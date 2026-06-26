#pragma once

#include <memory>  // Owns processors through std::unique_ptr.
#include <vector>  // Stores the ordered processor chain.

#include "core/IFrameProcessor.hpp"  // Defines the processor interface run by the pipeline.

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
