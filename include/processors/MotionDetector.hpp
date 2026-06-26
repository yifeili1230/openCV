#pragma once

#include <opencv2/core.hpp>    // Provides cv::Mat, cv::Rect, and geometry types.
#include <opencv2/imgproc.hpp> // Provides grayscale conversion, blur, morphology, and contours.
#include <opencv2/video.hpp>   // Provides OpenCV video-processing declarations.

#include <algorithm>  // Sorts detections by bounding-box area.

#include "core/IFrameProcessor.hpp"  // Defines the frame processor interface.

namespace video_engine {

class MotionDetector : public IFrameProcessor {
public:
    explicit MotionDetector(int threshold = 25, int min_area = 500, int merge_padding = 24,
                            int max_detections = 1)
        : threshold_(threshold),
          min_area_(min_area),
          merge_padding_(merge_padding),
          max_detections_(max_detections) {}

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
        cv::dilate(diff, diff, cv::Mat(), cv::Point(-1, -1), 3);
        cv::morphologyEx(diff, diff, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13)));

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> boxes;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < min_area_) {
                continue;
            }
            cv::Rect bbox = cv::boundingRect(contour);
            boxes.push_back(bbox);
        }

        boxes = mergeBoxes(boxes, ctx.processed_frame.size());
        std::sort(boxes.begin(), boxes.end(), [](const cv::Rect& lhs, const cv::Rect& rhs) {
            return lhs.area() > rhs.area();
        });

        ctx.detections.clear();
        const int output_count = max_detections_ <= 0
                                     ? static_cast<int>(boxes.size())
                                     : std::min(max_detections_, static_cast<int>(boxes.size()));
        for (int i = 0; i < output_count; ++i) {
            ctx.detections.push_back(Detection{boxes[i], -1});
        }

        gray_prev_ = gray;
    }

    std::string name() const override {
        return "motion_detector";
    }

private:
    std::vector<cv::Rect> mergeBoxes(const std::vector<cv::Rect>& boxes, cv::Size frame_size) const {
        std::vector<cv::Rect> merged;
        const cv::Rect frame_bounds(0, 0, frame_size.width, frame_size.height);

        for (const auto& box : boxes) {
            cv::Rect expanded = expandBox(box, frame_bounds);
            bool absorbed = false;
            for (auto& existing : merged) {
                cv::Rect expanded_existing = expandBox(existing, frame_bounds);
                if ((expanded & expanded_existing).area() > 0) {
                    existing |= box;
                    absorbed = true;
                    break;
                }
            }

            if (!absorbed) {
                merged.push_back(box);
            }
        }

        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t i = 0; i < merged.size() && !changed; ++i) {
                for (size_t j = i + 1; j < merged.size(); ++j) {
                    cv::Rect first = expandBox(merged[i], frame_bounds);
                    cv::Rect second = expandBox(merged[j], frame_bounds);
                    if ((first & second).area() > 0) {
                        merged[i] |= merged[j];
                        merged.erase(merged.begin() + static_cast<long>(j));
                        changed = true;
                        break;
                    }
                }
            }
        }

        return merged;
    }

    cv::Rect expandBox(const cv::Rect& box, const cv::Rect& frame_bounds) const {
        cv::Rect expanded(box.x - merge_padding_, box.y - merge_padding_,
                          box.width + merge_padding_ * 2, box.height + merge_padding_ * 2);
        return expanded & frame_bounds;
    }

    int threshold_;
    int min_area_;
    int merge_padding_;
    int max_detections_;
    cv::Mat gray_prev_;
};

}  // namespace video_engine
