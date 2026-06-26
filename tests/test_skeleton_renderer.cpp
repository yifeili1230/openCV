#include <opencv2/core.hpp>  // Provides cv::Mat and cv::Point for synthetic pose data.

#include "core/FrameContext.hpp"          // Provides pose storage for the renderer.
#include "processors/SkeletonRenderer.hpp" // Provides the skeleton renderer under test.

int main() {
    video_engine::FrameContext ctx;
    ctx.processed_frame = cv::Mat::zeros(240, 320, CV_8UC3);

    video_engine::Pose pose;
    pose.keypoints.resize(video_engine::PoseEstimator::kKeypointCount);
    const int points[] = {0, 1, 2, 3, 4, 8, 9, 10};
    for (int index : points) {
        pose.keypoints[index].visible = true;
        pose.keypoints[index].confidence = 0.9F;
        pose.keypoints[index].point = cv::Point(80 + index * 8, 60 + index * 4);
    }
    ctx.poses.push_back(pose);

    video_engine::SkeletonRenderer renderer;
    renderer.process(ctx);

    return ctx.processed_frame.empty() ? 1 : 0;
}
