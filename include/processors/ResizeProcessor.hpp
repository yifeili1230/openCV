#pragma once

#include <opencv2/imgproc.hpp>

#include "core/IFrameProcessor.hpp"

namespace video_engine {

class ResizeProcessor : public IFrameProcessor {
public:
    explicit ResizeProcessor(int width = 640, int height = 480) : width_(width), height_(height) {}

    void process(FrameContext& ctx) override {
        if (ctx.raw_frame.empty()) {
            return;
        }
        cv::resize(ctx.raw_frame, ctx.processed_frame, cv::Size(width_, height_));
    }

    std::string name() const override {
        return "resize";
    }

private:
    int width_;
    int height_;
};

}  // namespace video_engine
