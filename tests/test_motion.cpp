#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "processors/MotionDetector.hpp"

int main() {
    cv::Mat frame1 = cv::Mat::zeros(240, 320, CV_8UC3);
    cv::Mat frame2 = cv::Mat::zeros(240, 320, CV_8UC3);
    cv::rectangle(frame2, cv::Rect(50, 50, 40, 40), cv::Scalar(255, 255, 255), -1);

    video_engine::FrameContext ctx;
    ctx.raw_frame = frame1;
    ctx.processed_frame = frame1.clone();
    video_engine::MotionDetector detector(25, 50);
    detector.process(ctx);

    ctx.raw_frame = frame2;
    ctx.processed_frame = frame2.clone();
    detector.process(ctx);

    if (ctx.detections.empty()) {
        std::cerr << "Expected motion detection to find a contour" << std::endl;
        return 1;
    }
    return 0;
}
