#pragma once

#include <array>  // Stores the fixed skeleton edge list.

#include <opencv2/imgproc.hpp>  // Provides drawing APIs for skeleton lines and keypoints.

#include "core/IFrameProcessor.hpp"    // Defines the frame processor interface.
#include "processors/PoseEstimator.hpp" // Provides the OpenPose keypoint count constant.

namespace video_engine {

class SkeletonRenderer : public IFrameProcessor {
public:
    void process(FrameContext& ctx) override {
        if (ctx.processed_frame.empty()) {
            return;
        }

        cv::Mat overlay = ctx.processed_frame.clone();
        for (const auto& pose : ctx.poses) {
            if (pose.keypoints.size() < PoseEstimator::kKeypointCount) {
                continue;
            }

            for (const auto& [first, second] : kSkeletonPairs) {
                const auto& a = pose.keypoints[first];
                const auto& b = pose.keypoints[second];
                if (a.visible && b.visible) {
                    cv::line(overlay, a.point, b.point, cv::Scalar(45, 210, 255), 3, cv::LINE_AA);
                }
            }

            for (size_t i = 0; i < pose.keypoints.size(); ++i) {
                const auto& keypoint = pose.keypoints[i];
                if (!keypoint.visible) {
                    continue;
                }
                const cv::Scalar color = i == 0 ? cv::Scalar(0, 80, 255) : cv::Scalar(40, 255, 90);
                cv::circle(overlay, keypoint.point, i == 0 ? 6 : 5, color, cv::FILLED, cv::LINE_AA);
                cv::circle(overlay, keypoint.point, i == 0 ? 8 : 7, cv::Scalar(20, 20, 20), 1,
                           cv::LINE_AA);
            }
        }

        ctx.processed_frame = overlay;
    }

    std::string name() const override {
        return "skeleton_renderer";
    }

private:
    static constexpr std::array<std::pair<int, int>, 14> kSkeletonPairs = {{
        {1, 2},   // neck to right shoulder
        {2, 3},   // right shoulder to right elbow
        {3, 4},   // right elbow to right wrist
        {1, 5},   // neck to left shoulder
        {5, 6},   // left shoulder to left elbow
        {6, 7},   // left elbow to left wrist
        {1, 8},   // neck to right hip
        {8, 9},   // right hip to right knee
        {9, 10},  // right knee to right ankle
        {1, 11},  // neck to left hip
        {11, 12}, // left hip to left knee
        {12, 13}, // left knee to left ankle
        {0, 1},   // head point to neck
        {8, 11},  // hips
    }};
};

}  // namespace video_engine
