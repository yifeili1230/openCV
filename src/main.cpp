#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "core/FrameContext.hpp"
#include "core/Pipeline.hpp"
#include "processors/MotionDetector.hpp"
#include "processors/ObjectTracker.hpp"
#include "processors/OverlayRenderer.hpp"
#include "processors/ResizeProcessor.hpp"
#include "sources/VideoFileSource.hpp"
#include "sources/WebcamSource.hpp"
#include "utils/Profiler.hpp"

namespace {

struct AppConfig {
    std::string source = "webcam";
    std::string pipeline = "motion";
    std::string config_path = "configs/motion.yaml";
    bool display = true;
    bool save_output = false;
    std::string save_path = "output/output.mp4";
    int width = 640;
    int height = 480;
    int threshold = 25;
    int min_area = 500;
    int max_distance = 80;
    int max_age = 8;
    int trail_length = 25;
};

void applyConfigFile(AppConfig& config, const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        return;
    }

    std::string line;
    while (std::getline(input, line)) {
        auto comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        if (line.empty()) {
            continue;
        }
        auto delimiter = line.find(':');
        if (delimiter == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, delimiter);
        std::string value = line.substr(delimiter + 1);
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "source") {
            config.source = value;
        } else if (key == "width") {
            config.width = std::stoi(value);
        } else if (key == "height") {
            config.height = std::stoi(value);
        } else if (key == "display") {
            config.display = value == "true";
        } else if (key == "save_output") {
            config.save_output = value == "true";
        } else if (key == "save_path") {
            config.save_path = value;
        } else if (key == "threshold") {
            config.threshold = std::stoi(value);
        } else if (key == "min_area") {
            config.min_area = std::stoi(value);
        } else if (key == "max_distance") {
            config.max_distance = std::stoi(value);
        } else if (key == "max_age") {
            config.max_age = std::stoi(value);
        } else if (key == "trail_length") {
            config.trail_length = std::stoi(value);
        }
    }
}

AppConfig parseArgs(int argc, char** argv) {
    AppConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--source" && i + 1 < argc) {
            config.source = argv[++i];
        } else if (arg == "--pipeline" && i + 1 < argc) {
            config.pipeline = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config.config_path = argv[++i];
        } else if (arg == "--display") {
            config.display = true;
        } else if (arg == "--save" && i + 1 < argc) {
            config.save_output = true;
            config.save_path = argv[++i];
        }
    }
    applyConfigFile(config, config.config_path);
    return config;
}

std::unique_ptr<video_engine::IVideoSource> createSource(const AppConfig& config) {
    if (config.source == "webcam") {
        return std::make_unique<video_engine::WebcamSource>(0);
    }
    return std::make_unique<video_engine::VideoFileSource>(config.source);
}

std::unique_ptr<video_engine::Pipeline> buildPipeline(const AppConfig& config) {
    auto pipeline = std::make_unique<video_engine::Pipeline>();
    pipeline->addProcessor(std::make_unique<video_engine::ResizeProcessor>(config.width, config.height));
    pipeline->addProcessor(std::make_unique<video_engine::MotionDetector>(config.threshold, config.min_area));
    if (config.pipeline == "tracking") {
        pipeline->addProcessor(std::make_unique<video_engine::ObjectTracker>(config.max_distance, config.max_age, config.trail_length));
    }
    pipeline->addProcessor(std::make_unique<video_engine::OverlayRenderer>());
    return pipeline;
}

}  // namespace

int main(int argc, char** argv) {
    auto config = parseArgs(argc, argv);
    auto source = createSource(config);
    if (!source->open()) {
        std::cerr << "Failed to open source: " << config.source << std::endl;
        return 1;
    }

    cv::VideoWriter writer;
    if (config.save_output) {
        writer.open(config.save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0,
                    cv::Size(config.width, config.height));
    }

    auto pipeline = buildPipeline(config);
    video_engine::FrameContext ctx;

    size_t frame_index = 0;
    auto start_time = video_engine::Profiler::now();
    while (true) {
        cv::Mat frame;
        if (!source->read(frame)) {
            break;
        }
        ctx.raw_frame = frame;
        ctx.processed_frame.release();
        ctx.detections.clear();
        ctx.motion_points.clear();
        ctx.stage_names.clear();
        ctx.stage_latencies_ms.clear();
        ctx.trails.clear();

        auto stage_start = video_engine::Profiler::now();
        pipeline->run(ctx);
        auto stage_end = video_engine::Profiler::now();
        ctx.stage_names.push_back("pipeline");
        ctx.stage_latencies_ms.push_back(video_engine::Profiler::elapsedMs(stage_start, stage_end));

        if (ctx.processed_frame.empty()) {
            ctx.processed_frame = frame.clone();
        }

        if (config.display) {
            cv::imshow("Video Engine", ctx.processed_frame);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }

        if (writer.isOpened()) {
            writer.write(ctx.processed_frame);
        }

        ++frame_index;
        auto now = video_engine::Profiler::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double fps = elapsed > 0 ? frame_index / elapsed : 0.0;
        video_engine::Profiler::logFrameStats(frame_index, fps, ctx.stage_names, ctx.stage_latencies_ms);
    }

    cv::destroyAllWindows();
    return 0;
}
