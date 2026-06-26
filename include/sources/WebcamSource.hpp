#pragma once

#include <opencv2/videoio.hpp>  // Provides cv::VideoCapture for camera input.

#include "core/IVideoSource.hpp"  // Defines the common video source interface.

namespace video_engine {

class WebcamSource : public IVideoSource {
public:
    explicit WebcamSource(int device = 0) : device_(device) {}

    bool open() override {
        capture_.open(device_, cv::CAP_ANY);
        return capture_.isOpened();
    }

    bool read(cv::Mat& frame) override {
        return capture_.read(frame);
    }

    double fps() const override {
        return capture_.get(cv::CAP_PROP_FPS) > 0 ? capture_.get(cv::CAP_PROP_FPS) : 30.0;
    }

private:
    cv::VideoCapture capture_;
    int device_;
};

}  // namespace video_engine
