#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace video_engine {

class Profiler {
public:
    static std::chrono::steady_clock::time_point now() {
        return std::chrono::steady_clock::now();
    }

    static double elapsedMs(std::chrono::steady_clock::time_point start,
                            std::chrono::steady_clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    static void logFrameStats(size_t frame_index,
                              double fps,
                              const std::vector<std::string>& stage_names,
                              const std::vector<double>& latencies_ms) {
        std::ostringstream oss;
        oss << "[Frame " << frame_index << "] FPS: " << std::fixed << std::setprecision(1) << fps;
        for (size_t i = 0; i < stage_names.size() && i < latencies_ms.size(); ++i) {
            oss << " | " << stage_names[i] << ": " << std::setprecision(2) << latencies_ms[i] << " ms";
        }
        std::cout << oss.str() << std::endl;
    }
};

}  // namespace video_engine
