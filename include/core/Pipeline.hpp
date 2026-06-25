#pragma once

#include <memory>
#include <vector>

#include "core/IFrameProcessor.hpp"

namespace video_engine {

class Pipeline {
public:
    void addProcessor(std::unique_ptr<IFrameProcessor> processor) {
        processors_.push_back(std::move(processor));
    }

    void run(FrameContext& ctx) {
        for (const auto& processor : processors_) {
            processor->process(ctx);
        }
    }

private:
    std::vector<std::unique_ptr<IFrameProcessor>> processors_;
};

}  // namespace video_engine
