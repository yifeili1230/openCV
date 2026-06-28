#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "analytics/IPoseAnalyzer.hpp"

namespace video_engine {

struct SquatAnalyzerConfig {
    double minimum_joint_confidence = 0.12;
    double standing_angle_degrees = 160.0;
    double descent_start_angle_degrees = 155.0;
    double bottom_angle_degrees = 100.0;
    double bottom_exit_angle_degrees = 105.0;
    double minimum_normalized_speed_per_second = 0.02;
    double smoothing_alpha = 0.35;
};

class SquatAnalyzer : public IPoseAnalyzer {
public:
    explicit SquatAnalyzer(SquatAnalyzerConfig config = {}) : config_(config) {
        if (config_.smoothing_alpha <= 0.0 || config_.smoothing_alpha > 1.0) {
            throw std::invalid_argument("Squat smoothing alpha must be in (0, 1].");
        }
        if (!(config_.bottom_angle_degrees < config_.bottom_exit_angle_degrees &&
              config_.bottom_exit_angle_degrees < config_.descent_start_angle_degrees &&
              config_.descent_start_angle_degrees < config_.standing_angle_degrees)) {
            throw std::invalid_argument(
                "Squat angles must satisfy bottom < bottom exit < descent start < standing.");
        }
    }

    PoseAnalysisResult analyze(const Pose* pose, std::uint64_t frame_id,
                               double source_time_seconds) override {
        PoseAnalysisResult result;
        result.frame_id = frame_id;
        result.source_time_seconds = source_time_seconds;
        result.phase = phase_;
        result.completed_reps = static_cast<int>(completed_reps_.size());

        const SideObservation observation = selectSide(pose);
        if (!observation.valid) {
            ++invalid_frames_;
            previous_position_valid_ = false;
            return result;
        }

        ++valid_frames_;
        result.valid = true;
        result.observed_side = observation.name;

        const double raw_knee_angle =
            angleDegrees(observation.hip, observation.knee, observation.ankle);
        const double raw_hip_angle = observation.shoulder_valid
                                         ? angleDegrees(observation.shoulder, observation.hip,
                                                        observation.knee)
                                         : 0.0;

        if (!smoothed_values_valid_) {
            smoothed_knee_angle_ = raw_knee_angle;
            smoothed_hip_y_ = observation.hip.y;
            smoothed_values_valid_ = true;
        } else {
            smoothed_knee_angle_ =
                smooth(smoothed_knee_angle_, raw_knee_angle, config_.smoothing_alpha);
            smoothed_hip_y_ =
                smooth(smoothed_hip_y_, observation.hip.y, config_.smoothing_alpha);
        }

        double normalized_vertical_speed = 0.0;
        if (previous_position_valid_ && source_time_seconds > previous_source_time_seconds_) {
            const double delta_time = source_time_seconds - previous_source_time_seconds_;
            normalized_vertical_speed =
                (smoothed_hip_y_ - previous_smoothed_hip_y_) / observation.body_scale /
                delta_time;
        }

        const double angle_delta =
            previous_angle_valid_ ? smoothed_knee_angle_ - previous_knee_angle_ : 0.0;
        updateState(source_time_seconds, smoothed_knee_angle_, angle_delta,
                    normalized_vertical_speed, result);

        result.phase = phase_;
        result.completed_reps = static_cast<int>(completed_reps_.size());
        result.knee_angle_degrees = smoothed_knee_angle_;
        result.hip_angle_degrees = raw_hip_angle;
        result.hip_angle_valid = observation.shoulder_valid;
        result.normalized_vertical_speed_per_second = normalized_vertical_speed;

        previous_smoothed_hip_y_ = smoothed_hip_y_;
        previous_source_time_seconds_ = source_time_seconds;
        previous_position_valid_ = true;
        previous_knee_angle_ = smoothed_knee_angle_;
        previous_angle_valid_ = true;
        return result;
    }

    const std::vector<SquatRepSummary>& completedReps() const override {
        return completed_reps_;
    }

    std::size_t validFrameCount() const override {
        return valid_frames_;
    }

    std::size_t invalidFrameCount() const override {
        return invalid_frames_;
    }

    static double angleDegrees(const Point2D& first, const Point2D& vertex,
                               const Point2D& third) {
        const double first_x = static_cast<double>(first.x) - vertex.x;
        const double first_y = static_cast<double>(first.y) - vertex.y;
        const double third_x = static_cast<double>(third.x) - vertex.x;
        const double third_y = static_cast<double>(third.y) - vertex.y;
        const double first_length = std::hypot(first_x, first_y);
        const double third_length = std::hypot(third_x, third_y);
        if (first_length <= std::numeric_limits<double>::epsilon() ||
            third_length <= std::numeric_limits<double>::epsilon()) {
            return 0.0;
        }
        const double cosine = std::clamp(
            (first_x * third_x + first_y * third_y) / (first_length * third_length),
            -1.0, 1.0);
        constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
        return std::acos(cosine) * kRadiansToDegrees;
    }

private:
    struct SideObservation {
        bool valid = false;
        bool shoulder_valid = false;
        std::string name = "none";
        Point2D shoulder;
        Point2D hip;
        Point2D knee;
        Point2D ankle;
        double confidence = 0.0;
        double body_scale = 1.0;
    };

    SideObservation makeSide(const Pose* pose, JointId shoulder_id, JointId hip_id,
                             JointId knee_id, JointId ankle_id,
                             const std::string& name) const {
        SideObservation side;
        side.name = name;
        if (pose == nullptr || !pose->valid) {
            return side;
        }

        const auto& hip = pose->joint(hip_id);
        const auto& knee = pose->joint(knee_id);
        const auto& ankle = pose->joint(ankle_id);
        const auto& shoulder = pose->joint(shoulder_id);
        const auto usable = [this](const PoseKeypoint& point) {
            return point.valid && point.confidence >= config_.minimum_joint_confidence;
        };
        if (!usable(hip) || !usable(knee) || !usable(ankle)) {
            return side;
        }

        side.hip = hip.position_2d_pixels;
        side.knee = knee.position_2d_pixels;
        side.ankle = ankle.position_2d_pixels;
        side.confidence = std::min({static_cast<double>(hip.confidence),
                                    static_cast<double>(knee.confidence),
                                    static_cast<double>(ankle.confidence)});
        side.body_scale = std::hypot(static_cast<double>(side.hip.x - side.knee.x),
                                     static_cast<double>(side.hip.y - side.knee.y));
        if (side.body_scale <= 1.0) {
            return side;
        }

        side.shoulder_valid = usable(shoulder);
        if (side.shoulder_valid) {
            side.shoulder = shoulder.position_2d_pixels;
        }
        side.valid = true;
        return side;
    }

    SideObservation selectSide(const Pose* pose) const {
        const auto right = makeSide(pose, JointId::RightShoulder, JointId::RightHip,
                                    JointId::RightKnee, JointId::RightAnkle, "right");
        const auto left = makeSide(pose, JointId::LeftShoulder, JointId::LeftHip,
                                   JointId::LeftKnee, JointId::LeftAnkle, "left");
        if (!right.valid) {
            return left;
        }
        if (!left.valid) {
            return right;
        }
        return right.confidence >= left.confidence ? right : left;
    }

    static double smooth(double previous, double current, double alpha) {
        return alpha * current + (1.0 - alpha) * previous;
    }

    void beginRep(double source_time_seconds, double knee_angle) {
        rep_in_progress_ = true;
        rep_start_time_seconds_ = source_time_seconds;
        bottom_time_seconds_ = source_time_seconds;
        minimum_knee_angle_ = knee_angle;
        speed_sum_ = 0.0;
        speed_samples_ = 0;
        peak_speed_ = 0.0;
    }

    void recordSpeed(double speed) {
        if (!rep_in_progress_) {
            return;
        }
        const double magnitude = std::abs(speed);
        speed_sum_ += magnitude;
        ++speed_samples_;
        peak_speed_ = std::max(peak_speed_, magnitude);
    }

    void updateState(double source_time_seconds, double knee_angle, double angle_delta,
                     double normalized_vertical_speed, PoseAnalysisResult& result) {
        const bool moving_down =
            normalized_vertical_speed > config_.minimum_normalized_speed_per_second ||
            angle_delta < -0.5;
        const bool moving_up =
            normalized_vertical_speed < -config_.minimum_normalized_speed_per_second ||
            angle_delta > 0.5;

        switch (phase_) {
            case SquatPhase::Unknown:
                if (knee_angle >= config_.standing_angle_degrees) {
                    phase_ = SquatPhase::Standing;
                }
                break;
            case SquatPhase::Standing:
                if (knee_angle < config_.descent_start_angle_degrees && moving_down) {
                    phase_ = SquatPhase::Descending;
                    beginRep(source_time_seconds, knee_angle);
                }
                break;
            case SquatPhase::Descending:
                minimum_knee_angle_ = std::min(minimum_knee_angle_, knee_angle);
                recordSpeed(normalized_vertical_speed);
                if (knee_angle <= config_.bottom_angle_degrees) {
                    phase_ = SquatPhase::Bottom;
                    bottom_time_seconds_ = source_time_seconds;
                }
                break;
            case SquatPhase::Bottom:
                minimum_knee_angle_ = std::min(minimum_knee_angle_, knee_angle);
                recordSpeed(normalized_vertical_speed);
                if (knee_angle >= config_.bottom_exit_angle_degrees && moving_up) {
                    phase_ = SquatPhase::Ascending;
                }
                break;
            case SquatPhase::Ascending:
                minimum_knee_angle_ = std::min(minimum_knee_angle_, knee_angle);
                recordSpeed(normalized_vertical_speed);
                if (knee_angle >= config_.standing_angle_degrees) {
                    SquatRepSummary rep;
                    rep.rep_index = static_cast<int>(completed_reps_.size()) + 1;
                    rep.start_time_seconds = rep_start_time_seconds_;
                    rep.end_time_seconds = source_time_seconds;
                    rep.descent_time_seconds =
                        std::max(0.0, bottom_time_seconds_ - rep_start_time_seconds_);
                    rep.ascent_time_seconds =
                        std::max(0.0, source_time_seconds - bottom_time_seconds_);
                    rep.minimum_knee_angle_degrees = minimum_knee_angle_;
                    rep.average_normalized_speed_per_second =
                        speed_samples_ > 0 ? speed_sum_ / speed_samples_ : 0.0;
                    rep.peak_normalized_speed_per_second = peak_speed_;
                    completed_reps_.push_back(rep);
                    result.completed_rep = rep;
                    rep_in_progress_ = false;
                    phase_ = SquatPhase::Standing;
                }
                break;
        }
    }

    SquatAnalyzerConfig config_;
    SquatPhase phase_ = SquatPhase::Unknown;
    std::vector<SquatRepSummary> completed_reps_;
    std::size_t valid_frames_ = 0;
    std::size_t invalid_frames_ = 0;
    bool smoothed_values_valid_ = false;
    bool previous_position_valid_ = false;
    bool previous_angle_valid_ = false;
    bool rep_in_progress_ = false;
    double smoothed_knee_angle_ = 0.0;
    double smoothed_hip_y_ = 0.0;
    double previous_smoothed_hip_y_ = 0.0;
    double previous_source_time_seconds_ = 0.0;
    double previous_knee_angle_ = 0.0;
    double rep_start_time_seconds_ = 0.0;
    double bottom_time_seconds_ = 0.0;
    double minimum_knee_angle_ = 180.0;
    double speed_sum_ = 0.0;
    std::size_t speed_samples_ = 0;
    double peak_speed_ = 0.0;
};

}  // namespace video_engine
