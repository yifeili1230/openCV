#pragma once

#include <opencv2/core.hpp>

namespace video_engine {

class IVideoSource {
public:
    virtual ~IVideoSource() = default;
    virtual bool open() = 0;
    virtual bool read(cv::Mat& frame) = 0;
    virtual double fps() const = 0;
};

}  // namespace video_engine
