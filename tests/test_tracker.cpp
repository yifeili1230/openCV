#include <iostream>  // Prints test failure messages.
#include <set>       // Verifies that produced track IDs are unique.

#include <opencv2/core.hpp>  // Provides cv::Rect for synthetic detections.

#include "core/FrameContext.hpp"        // Provides Detection and FrameContext test data.
#include "processors/ObjectTracker.hpp" // Provides the tracker under test.

int main() {
    video_engine::ObjectTracker tracker(80, 2, 10);
    video_engine::FrameContext ctx;

    ctx.detections = {
        video_engine::Detection{cv::Rect(10, 10, 20, 20), -1},
        video_engine::Detection{cv::Rect(150, 10, 20, 20), -1},
    };
    tracker.process(ctx);

    ctx.detections = {
        video_engine::Detection{cv::Rect(12, 10, 20, 20), -1},
        video_engine::Detection{cv::Rect(18, 10, 20, 20), -1},
    };
    tracker.process(ctx);

    std::set<int> ids;
    for (const auto& detection : ctx.detections) {
        ids.insert(detection.id);
    }

    if (ids.size() != ctx.detections.size()) {
        std::cerr << "Expected one unique track id per detection" << std::endl;
        return 1;
    }

    ctx.detections.clear();
    tracker.process(ctx);
    tracker.process(ctx);
    tracker.process(ctx);

    return 0;
}
