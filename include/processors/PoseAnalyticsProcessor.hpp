#pragma once

#include <memory>

#include "analytics/IPoseAnalyzer.hpp"
#include "core/IFrameProcessor.hpp"

namespace video_engine {

class PoseAnalyticsProcessor : public IFrameProcessor {
public:
    explicit PoseAnalyticsProcessor(std::shared_ptr<IPoseAnalyzer> analyzer)
        : analyzer_(std::move(analyzer)) {}

    void process(FrameContext& ctx) override {
        const Pose* pose = nullptr;
        if (ctx.pose_measurement_valid && !ctx.poses.empty()) {
            pose = &ctx.poses.front();
        }
        ctx.pose_analysis =
            analyzer_->analyze(pose, ctx.frame_id, ctx.source_time_seconds);
        ctx.completed_rep_events.clear();
        if (ctx.pose_analysis->completed_rep.has_value()) {
            ctx.completed_rep_events.push_back(*ctx.pose_analysis->completed_rep);
        }
    }

    std::string name() const override {
        return "pose_analytics";
    }

private:
    std::shared_ptr<IPoseAnalyzer> analyzer_;
};

}  // namespace video_engine
