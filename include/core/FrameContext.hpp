#pragma once

#include <chrono>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace video_engine {

struct Detection {
    cv::Rect bbox;
    int id = -1;
};

struct FrameContext {
    cv::Mat raw_frame;
    cv::Mat processed_frame;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<Detection> detections;
    std::vector<cv::Point> motion_points;
    std::vector<std::string> stage_names;
    std::vector<double> stage_latencies_ms;
    std::vector<std::vector<cv::Point>> trails;
    std::string metadata;
};

}  // namespace video_engine
