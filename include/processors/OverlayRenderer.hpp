#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "core/IFrameProcessor.hpp"

namespace video_engine {

class OverlayRenderer : public IFrameProcessor {
public:
    void process(FrameContext& ctx) override {
        if (ctx.processed_frame.empty()) {
            return;
        }

        cv::Mat overlay = ctx.processed_frame.clone();
        for (const auto& detection : ctx.detections) {
            cv::rectangle(overlay, detection.bbox, cv::Scalar(0, 255, 0), 2);
            cv::putText(overlay, std::to_string(detection.id),
                        cv::Point(detection.bbox.x, detection.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        for (const auto& trail : ctx.trails) {
            if (trail.size() < 2) {
                continue;
            }
            for (size_t i = 1; i < trail.size(); ++i) {
                cv::line(overlay, trail[i - 1], trail[i], cv::Scalar(255, 0, 0), 2);
            }
        }

        ctx.processed_frame = overlay;
    }

    std::string name() const override {
        return "overlay";
    }
};

}  // namespace video_engine
