#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include <opencv2/imgproc.hpp>

#include "analytics/AnalyticsTypes.hpp"
#include "core/IFrameProcessor.hpp"

namespace video_engine {

class PoseAnalyticsRenderer : public IFrameProcessor {
public:
    explicit PoseAnalyticsRenderer(int panel_width = 300) : panel_width_(panel_width) {}

    void process(FrameContext& ctx) override {
        if (ctx.processed_frame.empty()) {
            return;
        }

        cv::Mat canvas(ctx.processed_frame.rows, ctx.processed_frame.cols + panel_width_,
                       ctx.processed_frame.type(), cv::Scalar(20, 23, 29));
        ctx.processed_frame.copyTo(
            canvas(cv::Rect(0, 0, ctx.processed_frame.cols, ctx.processed_frame.rows)));
        cv::Mat panel = canvas(
            cv::Rect(ctx.processed_frame.cols, 0, panel_width_, ctx.processed_frame.rows));

        drawText(panel, "SQUAT ANALYSIS", 32, 0.70, cv::Scalar(245, 245, 245), 2);
        cv::line(panel, cv::Point(20, 48), cv::Point(panel_width_ - 20, 48),
                 cv::Scalar(65, 70, 80), 1, cv::LINE_AA);

        if (!ctx.pose_analysis.has_value()) {
            drawText(panel, "Waiting for analysis", 86, 0.55, cv::Scalar(150, 155, 165), 1);
            ctx.processed_frame = canvas;
            return;
        }

        const auto& analysis = *ctx.pose_analysis;
        const cv::Scalar phase_color = phaseColor(analysis.phase);
        drawLabel(panel, "PHASE", 82);
        drawText(panel, squatPhaseName(analysis.phase), 110, 0.72, phase_color, 2);

        drawLabel(panel, "REPETITIONS", 154);
        drawText(panel, std::to_string(analysis.completed_reps), 192, 1.15,
                 cv::Scalar(255, 220, 80), 2);

        drawLabel(panel, "LIVE METRICS", 234);
        if (analysis.valid) {
            drawText(panel, "Knee angle  " + format(analysis.knee_angle_degrees, 1) + " deg",
                     264, 0.55, cv::Scalar(235, 235, 235), 1);
            const std::string hip_value = analysis.hip_angle_valid
                                              ? format(analysis.hip_angle_degrees, 1) + " deg"
                                              : "n/a";
            drawText(panel, "Hip angle   " + hip_value, 294, 0.55,
                     cv::Scalar(235, 235, 235), 1);
            drawText(panel,
                     "Vertical speed " +
                         format(analysis.normalized_vertical_speed_per_second, 2) + " body/s",
                     324, 0.50, cv::Scalar(235, 235, 235), 1);
            drawText(panel, "Observed side  " + analysis.observed_side, 354, 0.50,
                     cv::Scalar(180, 190, 205), 1);
        } else {
            drawText(panel, "Pose confidence too low", 270, 0.50,
                     cv::Scalar(80, 170, 255), 1);
            drawText(panel, "State is paused", 300, 0.50, cv::Scalar(180, 190, 205), 1);
        }

        if (analysis.completed_rep.has_value()) {
            const auto& rep = *analysis.completed_rep;
            drawText(panel, "REP " + std::to_string(rep.rep_index) + " COMPLETE", 410, 0.62,
                     cv::Scalar(80, 230, 130), 2);
            drawText(panel, "Min knee  " + format(rep.minimum_knee_angle_degrees, 1) + " deg",
                     440, 0.50, cv::Scalar(220, 225, 230), 1);
            drawText(panel, "Down / up  " + format(rep.descent_time_seconds, 2) + " / " +
                                format(rep.ascent_time_seconds, 2) + " s",
                     466, 0.48, cv::Scalar(220, 225, 230), 1);
        } else {
            drawText(panel, "ESC: finish and save summary", panel.rows - 22, 0.42,
                     cv::Scalar(125, 135, 150), 1);
        }

        ctx.processed_frame = canvas;
    }

    std::string name() const override {
        return "pose_analytics_renderer";
    }

private:
    static std::string format(double value, int precision) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(precision) << value;
        return stream.str();
    }

    static cv::Scalar phaseColor(SquatPhase phase) {
        switch (phase) {
            case SquatPhase::Descending:
                return cv::Scalar(80, 190, 255);
            case SquatPhase::Bottom:
                return cv::Scalar(90, 90, 255);
            case SquatPhase::Ascending:
                return cv::Scalar(100, 230, 140);
            case SquatPhase::Standing:
                return cv::Scalar(255, 220, 80);
            case SquatPhase::Unknown:
            default:
                return cv::Scalar(160, 165, 175);
        }
    }

    static void drawLabel(cv::Mat& panel, const std::string& text, int y) {
        drawText(panel, text, y, 0.43, cv::Scalar(130, 140, 155), 1);
    }

    static void drawText(cv::Mat& panel, const std::string& text, int y, double scale,
                         const cv::Scalar& color, int thickness) {
        cv::putText(panel, text, cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, scale, color,
                    thickness, cv::LINE_AA);
    }

    int panel_width_;
};

}  // namespace video_engine
