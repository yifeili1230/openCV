#include <cmath>
#include <iostream>
#include <vector>

#include "analytics/SquatAnalyzer.hpp"
#include "analytics/SquatSummaryWriter.hpp"
#include "processors/PoseAnalyticsRenderer.hpp"

namespace {

video_engine::Pose makePose(double knee_angle_degrees, double hip_y,
                            float confidence = 0.95F) {
    video_engine::Pose pose;
    pose.valid = true;

    constexpr double kPi = 3.14159265358979323846;
    const double radians = (-90.0 + knee_angle_degrees) * kPi / 180.0;
    const video_engine::Point2D hip{100.0F, static_cast<float>(hip_y)};
    const video_engine::Point2D knee{100.0F, static_cast<float>(hip_y + 50.0)};
    const video_engine::Point2D ankle{
        static_cast<float>(knee.x + 50.0 * std::cos(radians)),
        static_cast<float>(knee.y + 50.0 * std::sin(radians)),
    };
    const video_engine::Point2D shoulder{100.0F, static_cast<float>(hip_y - 50.0)};

    for (const auto [joint, point] :
         std::vector<std::pair<video_engine::JointId, video_engine::Point2D>>{
             {video_engine::JointId::RightShoulder, shoulder},
             {video_engine::JointId::RightHip, hip},
             {video_engine::JointId::RightKnee, knee},
             {video_engine::JointId::RightAnkle, ankle},
         }) {
        auto& keypoint = pose.joint(joint);
        keypoint.position_2d_pixels = point;
        keypoint.confidence = confidence;
        keypoint.valid = true;
    }
    return pose;
}

}  // namespace

int main() {
    const auto right_angle = video_engine::SquatAnalyzer::angleDegrees(
        video_engine::Point2D{0.0F, 1.0F}, video_engine::Point2D{0.0F, 0.0F},
        video_engine::Point2D{1.0F, 0.0F});
    if (std::abs(right_angle - 90.0) > 0.001) {
        std::cerr << "2D joint angle calculation is incorrect" << std::endl;
        return 1;
    }

    video_engine::SquatAnalyzerConfig config;
    config.smoothing_alpha = 1.0;
    video_engine::SquatAnalyzer analyzer(config);
    const std::vector<double> angles{
        170.0, 166.0, 150.0, 130.0, 110.0, 95.0, 98.0, 110.0, 130.0, 150.0, 165.0,
    };
    const std::vector<double> hip_y{
        50.0, 50.0, 53.0, 58.0, 64.0, 70.0, 70.0, 67.0, 61.0, 55.0, 50.0,
    };

    video_engine::PoseAnalysisResult result;
    for (std::size_t index = 0; index < angles.size(); ++index) {
        auto pose = makePose(angles[index], hip_y[index]);
        result = analyzer.analyze(&pose, index + 1, index * 0.1);
    }

    if (analyzer.completedReps().size() != 1 || result.completed_reps != 1 ||
        !result.completed_rep.has_value()) {
        std::cerr << "Complete standing-descending-bottom-ascending cycle was not counted"
                  << std::endl;
        return 1;
    }

    const auto& rep = analyzer.completedReps().front();
    if (rep.minimum_knee_angle_degrees > 100.0 || rep.descent_time_seconds <= 0.0 ||
        rep.ascent_time_seconds <= 0.0 || rep.peak_normalized_speed_per_second <= 0.0) {
        std::cerr << "Completed rep metrics are incomplete" << std::endl;
        return 1;
    }

    auto invalid_pose = makePose(90.0, 70.0, 0.01F);
    const auto invalid = analyzer.analyze(&invalid_pose, 20, 2.0);
    if (invalid.valid || analyzer.invalidFrameCount() != 1 ||
        analyzer.completedReps().size() != 1) {
        std::cerr << "Low-confidence observation corrupted squat state" << std::endl;
        return 1;
    }

    video_engine::SquatAnalyzer incomplete_analyzer(config);
    const std::vector<double> incomplete_angles{170.0, 150.0, 120.0, 110.0, 150.0, 170.0};
    for (std::size_t index = 0; index < incomplete_angles.size(); ++index) {
        auto pose = makePose(incomplete_angles[index], 50.0 + index);
        incomplete_analyzer.analyze(&pose, index + 1, index * 0.1);
    }
    if (!incomplete_analyzer.completedReps().empty()) {
        std::cerr << "Squat that never reached the bottom threshold was counted" << std::endl;
        return 1;
    }

    video_engine::SquatSessionSummary summary;
    summary.source = "video_source/IMG_1389.mov";
    summary.processed_frames = angles.size();
    summary.valid_analysis_frames = analyzer.validFrameCount();
    summary.invalid_analysis_frames = analyzer.invalidFrameCount();
    summary.reps = analyzer.completedReps();
    const auto summary_path =
        video_engine::SquatSummaryWriter::outputPathForSource(summary.source, "output");
    const auto json = video_engine::SquatSummaryWriter::toJson(summary);
    if (summary_path != std::filesystem::path("output/IMG_1389.json") ||
        json.find("\"total_reps\": 1") == std::string::npos ||
        json.find("\"minimum_knee_angle_degrees\"") == std::string::npos) {
        std::cerr << "Input-named squat JSON summary is incorrect" << std::endl;
        return 1;
    }

    video_engine::FrameContext context;
    context.processed_frame = cv::Mat::zeros(480, 640, CV_8UC3);
    context.pose_analysis = result;
    video_engine::PoseAnalyticsRenderer renderer;
    renderer.process(context);
    if (context.processed_frame.cols != 940 || context.processed_frame.rows != 480) {
        std::cerr << "Live analytics panel did not preserve video and add side panel"
                  << std::endl;
        return 1;
    }

    return 0;
}
