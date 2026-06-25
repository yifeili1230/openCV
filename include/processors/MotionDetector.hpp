#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include "core/IFrameProcessor.hpp"

namespace video_engine {

class MotionDetector : public IFrameProcessor {
public:
    explicit MotionDetector(int threshold = 25, int min_area = 500)
        : threshold_(threshold), min_area_(min_area) {}

    void process(FrameContext& ctx) override {
        if (ctx.processed_frame.empty()) {
            return;
        }

        if (gray_prev_.empty()) {
            cv::cvtColor(ctx.processed_frame, gray_prev_, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray_prev_, gray_prev_, cv::Size(5, 5), 0);
            return;
        }

        cv::Mat gray;
        cv::cvtColor(ctx.processed_frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        cv::Mat diff;
        cv::absdiff(gray_prev_, gray, diff);
        cv::threshold(diff, diff, threshold_, 255, cv::THRESH_BINARY);
        cv::erode(diff, diff, cv::Mat(), cv::Point(-1, -1), 1);
        cv::dilate(diff, diff, cv::Mat(), cv::Point(-1, -1), 2);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        ctx.detections.clear();
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < min_area_) {
                continue;
            }
            cv::Rect bbox = cv::boundingRect(contour);
            ctx.detections.push_back(Detection{bbox, -1});
        }

        gray_prev_ = gray;
    }

    std::string name() const override {
        return "motion_detector";
    }

private:
    int threshold_;
    int min_area_;
    cv::Mat gray_prev_;
};

}  // namespace video_engine
