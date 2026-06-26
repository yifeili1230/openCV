#pragma once

#include <string>  // Stores the input video file path.

#include <opencv2/videoio.hpp>  // Provides cv::VideoCapture for video file input.

#include "core/IVideoSource.hpp"  // Defines the common video source interface.

namespace video_engine {

class VideoFileSource : public IVideoSource {
public:
    explicit VideoFileSource(std::string path) : path_(std::move(path)) {}

    bool open() override {
        capture_.open(path_);
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
    std::string path_;
};

}  // namespace video_engine
