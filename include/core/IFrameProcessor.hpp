#pragma once

#include "core/FrameContext.hpp"

namespace video_engine {

class IFrameProcessor {
public:
    virtual ~IFrameProcessor() = default;
    virtual void process(FrameContext& ctx) = 0;
    virtual std::string name() const = 0;
};

}  // namespace video_engine
