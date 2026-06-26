#pragma once

#include <chrono>  // Stores timestamps for frame-level timing.
#include <string>  // Stores metadata and profiling stage names.
#include <vector>  // Stores detections, poses, trails, and timing arrays.

#include <opencv2/core.hpp>  // Provides cv::Mat, cv::Rect, and cv::Point.

namespace video_engine {

struct Detection {
    cv::Rect bbox;
    int id = -1;
};

struct PoseKeypoint {
    cv::Point point;
    float confidence = 0.0F;
    bool visible = false;
};

struct Pose {
    std::vector<PoseKeypoint> keypoints;
};

struct FrameContext {
    cv::Mat raw_frame;
    cv::Mat processed_frame;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<Detection> detections;
    std::vector<Pose> poses;
    std::vector<cv::Point> motion_points;
    std::vector<std::string> stage_names;
    std::vector<double> stage_latencies_ms;
    std::vector<std::vector<cv::Point>> trails;
    std::string metadata;
};

}  // namespace video_engine
