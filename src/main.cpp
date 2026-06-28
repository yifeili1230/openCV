#include <chrono>    // Provides timing utilities for FPS and pipeline latency measurement.
#include <csignal>   // Handles Ctrl+C as a graceful request to finish and save outputs.
#include <filesystem> // Provides output directory creation and file existence checks.
#include <fstream>   // Provides file input for reading configs/*.yaml.
#include <iostream>  // Provides standard output streams for error messages.
#include <memory>    // Provides std::unique_ptr for owned source, pipeline, and processor objects.
#include <stdexcept> // Provides runtime errors for invalid inference platform selections.
#include <string>    // Provides std::string for paths, pipeline names, and backend names.
#include <vector>    // Provides std::vector for stages, detections, keypoints, and trails.

#include <opencv2/highgui.hpp>  // Provides OpenCV window display APIs such as cv::imshow and cv::waitKey.
#include <opencv2/imgproc.hpp>  // Provides OpenCV image processing APIs such as resize and drawing.
#include <opencv2/videoio.hpp>  // Provides OpenCV video capture/output APIs such as cv::VideoWriter.

#include "analytics/IPoseAnalyzer.hpp"        // Defines the model-independent analytics interface.
#include "analytics/SquatAnalyzer.hpp"        // Implements mathematical squat analysis.
#include "analytics/SquatSummaryWriter.hpp"   // Writes the final input-named JSON summary.
#include "core/FrameContext.hpp"             // Defines the per-frame data object shared across processors.
#include "core/FrameTimeline.hpp"            // Assigns monotonic frame IDs and source timestamps.
#include "core/Pipeline.hpp"                 // Runs a sequence of IFrameProcessor stages.
#include "processors/MotionDetector.hpp"     // Detects moving regions with frame differencing.
#include "processors/ObjectTracker.hpp"      // Tracks detections by matching bounding-box centroids.
#include "processors/OverlayRenderer.hpp"    // Draws detection boxes, IDs, and trails on the frame.
#include "processors/PoseAnalyticsProcessor.hpp" // Connects canonical poses to exercise analysis.
#include "processors/PoseAnalyticsRenderer.hpp"  // Draws live squat metrics beside the video.
#include "processors/PoseEstimator.hpp"      // Runs human-pose keypoint inference with OpenCV DNN.
#include "processors/ResizeProcessor.hpp"    // Resizes each input frame to the configured dimensions.
#include "processors/SkeletonRenderer.hpp"   // Draws pose keypoints and skeleton connections.
#include "sources/VideoFileSource.hpp"       // Provides video-file input.
#include "sources/WebcamSource.hpp"          // Provides webcam input.
#include "utils/Profiler.hpp"                // Logs frame timing, FPS, and stage latency.

namespace {  // Keeps helper types and functions private to this translation unit.

volatile std::sig_atomic_t stop_requested = 0;

void requestStop(int) {
    stop_requested = 1;
}

struct AppConfig {  // Stores all runtime configuration with defaults.
    std::string source = "webcam";  // Input source; "webcam" means camera, otherwise it is a file path.
    std::string pipeline = "pose";  // Pipeline mode; supported values are motion, tracking, and pose.
    std::string config_path = "configs/pose.yaml";  // Default config file path.
    bool display = true;  // Whether to show a live OpenCV window.
    bool save_output = true;  // Whether to save processed frames as a video.
    std::string save_path = "output/output";  // Output video prefix used for numbered MP4 files.
    std::string exercise = "none";  // Optional exercise analyzer; supported values are none and squat.
    std::string analysis_output_dir = "output";  // Directory for input-named analytics summaries.
    int width = 640;  // Processing frame width used by ResizeProcessor.
    int height = 480;  // Processing frame height used by ResizeProcessor.
    int threshold = 25;  // Motion binary threshold; higher values are less sensitive.
    int min_area = 500;  // Minimum contour area for keeping a motion detection.
    int merge_padding = 24;  // Pixel padding used when merging nearby motion boxes.
    int max_detections = 1;  // Maximum number of motion detections to keep; 1 keeps the largest target.
    int max_distance = 80;  // Maximum centroid distance for assigning a detection to an existing track.
    int max_age = 8;  // Number of missed frames before a track is removed.
    int trail_length = 25;  // Maximum number of historical points stored for each track trail.
    std::string pose_model = "models/pose_iter_440000.caffemodel";  // Pose model weights path.
    std::string pose_config = "models/pose_deploy_linevec.prototxt";  // Pose model network definition path.
    int pose_input_width = 368;  // Width of the tensor sent into the pose model.
    int pose_input_height = 368;  // Height of the tensor sent into the pose model.
    float pose_confidence = 0.12F;  // Minimum keypoint confidence required for rendering.
    std::string inference_platform = "manual";  // Optional profile that selects a matching DNN backend and target.
    std::string pose_backend = "opencv";  // OpenCV DNN backend, for example opencv, openvino, or cuda.
    std::string pose_target = "cpu";  // OpenCV DNN target device, for example cpu, opencl, cuda, or npu.
    double squat_standing_angle = 160.0;  // Knee angle that completes a return to standing.
    double squat_descent_start_angle = 155.0;  // Hysteresis threshold for leaving standing.
    double squat_bottom_angle = 100.0;  // Knee angle required to establish squat depth.
    double squat_bottom_exit_angle = 105.0;  // Hysteresis threshold for leaving the bottom.
    double squat_minimum_speed = 0.02;  // Minimum normalized vertical movement used for direction.
    double squat_smoothing_alpha = 0.35;  // Exponential smoothing weight for new observations.
};  // End of AppConfig.

void configureInferencePlatform(AppConfig& config, const std::string& platform) {  // Applies convenient hardware-specific inference defaults.
    if (platform.empty() || platform == "manual") {  // Keeps explicitly configured backend and target values.
        return;  // Leaves the detailed DNN settings unchanged.
    }  // End of manual configuration branch.
    if (platform == "cpu") {  // Selects the portable CPU profile.
        config.pose_backend = "opencv";  // Uses OpenCV's built-in DNN implementation.
        config.pose_target = "cpu";  // Runs network layers on the CPU.
        return;  // Finishes applying the CPU profile.
    }  // End of CPU profile branch.
    if (platform == "jetson" || platform == "orin-nano" ||
        platform == "jetson-orin-nano") {  // Selects NVIDIA Jetson Orin Nano GPU inference.
        config.pose_backend = "cuda";  // Uses OpenCV DNN's CUDA backend.
        config.pose_target = "cuda_fp16";  // Uses FP16 CUDA inference to reduce compute and memory traffic.
        return;  // Finishes applying the Jetson profile.
    }  // End of Jetson profile branch.
    throw std::runtime_error(  // Rejects misspelled profiles instead of unexpectedly running on the CPU.
        "Unsupported inference platform: " + platform +
        ". Use manual, cpu, jetson, or jetson-orin-nano.");
}  // End of configureInferencePlatform.

void applyConfigFile(AppConfig& config, const std::string& path) {  // Applies config-file values to AppConfig.
    std::ifstream input(path);  // Opens the config file.
    if (!input.is_open()) {  // Handles missing or unreadable config files.
        return;  // Keeps the current defaults and exits.
    }  // End of open failure handling.

    std::string line;  // Holds the current config line.
    while (std::getline(input, line)) {  // Reads the config file one line at a time.
        auto comment = line.find('#');  // Finds an inline YAML-style comment marker.
        if (comment != std::string::npos) {  // Checks whether the line contains a comment.
            line = line.substr(0, comment);  // Removes the comment portion.
        }  // End of comment trimming.
        if (line.empty()) {  // Skips empty lines.
            continue;  // Moves to the next line.
        }  // End of empty-line handling.
        auto delimiter = line.find(':');  // Finds the key/value delimiter.
        if (delimiter == std::string::npos) {  // Skips lines without a delimiter.
            continue;  // Moves to the next line.
        }  // End of malformed-line handling.
        std::string key = line.substr(0, delimiter);  // Extracts the key on the left side of ':'.
        std::string value = line.substr(delimiter + 1);  // Extracts the value on the right side of ':'.
        key.erase(0, key.find_first_not_of(" \t"));  // Trims leading whitespace from the key.
        key.erase(key.find_last_not_of(" \t") + 1);  // Trims trailing whitespace from the key.
        value.erase(0, value.find_first_not_of(" \t"));  // Trims leading whitespace from the value.
        value.erase(value.find_last_not_of(" \t") + 1);  // Trims trailing whitespace from the value.

        if (key == "source") {  // Configures the input source.
            config.source = value;  // Stores the source string.
        } else if (key == "width") {  // Configures processing width.
            config.width = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "height") {  // Configures processing height.
            config.height = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "display") {  // Configures live display.
            config.display = value == "true";  // Enables display only for the literal value "true".
        } else if (key == "save_output") {  // Configures output video saving.
            config.save_output = value == "true";  // Enables saving only for the literal value "true".
        } else if (key == "save_path") {  // Configures output video path.
            config.save_path = value;  // Stores the output path.
        } else if (key == "exercise") {  // Selects an optional exercise analyzer.
            config.exercise = value;  // Stores none or squat for validation after CLI overrides.
        } else if (key == "analysis_output_dir") {  // Configures summary output location.
            config.analysis_output_dir = value;  // Stores the analytics directory.
        } else if (key == "threshold") {  // Configures motion threshold.
            config.threshold = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "min_area") {  // Configures minimum motion contour area.
            config.min_area = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "merge_padding") {  // Configures motion box merge padding.
            config.merge_padding = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "max_detections") {  // Configures maximum motion detections.
            config.max_detections = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "max_distance") {  // Configures tracking match distance.
            config.max_distance = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "max_age") {  // Configures tracking expiration age.
            config.max_age = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "trail_length") {  // Configures rendered track trail length.
            config.trail_length = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "pose_model") {  // Configures pose model weights path.
            config.pose_model = value;  // Stores the model path.
        } else if (key == "pose_config") {  // Configures pose model definition path.
            config.pose_config = value;  // Stores the model config path.
        } else if (key == "pose_input_width") {  // Configures pose model input width.
            config.pose_input_width = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "pose_input_height") {  // Configures pose model input height.
            config.pose_input_height = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "pose_confidence") {  // Configures pose keypoint confidence threshold.
            config.pose_confidence = std::stof(value);  // Converts the value to a float.
        } else if (key == "inference_platform") {  // Configures a hardware-specific inference profile.
            config.inference_platform = value;  // Stores the profile name for application after all overrides.
        } else if (key == "pose_backend") {  // Configures the OpenCV DNN backend.
            config.pose_backend = value;  // Stores the backend name.
        } else if (key == "pose_target") {  // Configures the OpenCV DNN target device.
            config.pose_target = value;  // Stores the target device name.
        } else if (key == "squat_standing_angle") {
            config.squat_standing_angle = std::stod(value);
        } else if (key == "squat_descent_start_angle") {
            config.squat_descent_start_angle = std::stod(value);
        } else if (key == "squat_bottom_angle") {
            config.squat_bottom_angle = std::stod(value);
        } else if (key == "squat_bottom_exit_angle") {
            config.squat_bottom_exit_angle = std::stod(value);
        } else if (key == "squat_minimum_speed") {
            config.squat_minimum_speed = std::stod(value);
        } else if (key == "squat_smoothing_alpha") {
            config.squat_smoothing_alpha = std::stod(value);
        }  // End of config key dispatch.
    }  // End of config-file line loop.
}  // End of applyConfigFile.

AppConfig parseArgs(int argc, char** argv) {  // Parses CLI arguments and loads the selected config file.
    AppConfig config;  // Starts with default configuration.
    for (int i = 1; i < argc; ++i) {  // First pass only finds the requested config file.
        std::string arg = argv[i];  // Wraps the current argument in std::string for comparison.
        if (arg == "--config" && i + 1 < argc) {  // Handles --config before loading file settings.
            config.config_path = argv[++i];  // Consumes the next argument as the config path.
        }  // End of config-path pre-scan.
    }  // End of config-path pre-scan loop.
    applyConfigFile(config, config.config_path);  // Applies config-file values before CLI overrides.

    for (int i = 1; i < argc; ++i) {  // Iterates over user-provided command-line arguments.
        std::string arg = argv[i];  // Wraps the current argument in std::string for comparison.
        if (arg == "--source" && i + 1 < argc) {  // Handles --source when a value follows it.
            config.source = argv[++i];  // Consumes the next argument as the input source.
        } else if (arg == "--pipeline" && i + 1 < argc) {  // Handles --pipeline when a value follows it.
            config.pipeline = argv[++i];  // Consumes the next argument as the pipeline mode.
        } else if (arg == "--config" && i + 1 < argc) {  // Handles --config when a value follows it.
            config.config_path = argv[++i];  // Consumes the next argument as the config path.
        } else if (arg == "--display") {  // Handles explicit live-display request.
            config.display = true;  // Enables display.
        } else if (arg == "--no-display") {  // Handles headless operation over SSH or without a desktop session.
            config.display = false;  // Disables OpenCV window creation.
        } else if (arg == "--inference-platform" && i + 1 < argc) {  // Handles a named inference hardware profile.
            config.inference_platform = argv[++i];  // Consumes manual, cpu, or jetson.
        } else if (arg == "--save" && i + 1 < argc) {  // Handles output video saving.
            config.save_output = true;  // Enables saving.
            config.save_path = argv[++i];  // Consumes the next argument as the output path.
        } else if (arg == "--no-save") {  // Handles explicit output-video disabling.
            config.save_output = false;  // Avoids encoding overhead when only live inference is needed.
        } else if (arg == "--exercise" && i + 1 < argc) {  // Selects exercise analysis.
            config.exercise = argv[++i];  // Consumes none or squat.
        } else if (arg == "--analysis-output" && i + 1 < argc) {  // Selects summary directory.
            config.analysis_output_dir = argv[++i];  // Consumes the output directory.
        }  // End of CLI argument dispatch.
    }  // End of CLI argument loop.
    if (config.exercise != "none" && config.exercise != "squat") {
        throw std::runtime_error("Unsupported exercise: " + config.exercise +
                                 ". Use none or squat.");
    }
    if (config.exercise == "squat" && config.pipeline != "pose") {
        throw std::runtime_error("Squat analysis requires --pipeline pose.");
    }
    configureInferencePlatform(config, config.inference_platform);  // Resolves the final platform into backend and target settings.
    return config;  // Returns the final runtime configuration.
}  // End of parseArgs.

std::unique_ptr<video_engine::IVideoSource> createSource(const AppConfig& config) {  // Creates the video input source.
    if (config.source == "webcam") {  // Selects the default webcam source.
        return std::make_unique<video_engine::WebcamSource>(0);  // Opens camera index 0.
    }  // End of webcam branch.
    return std::make_unique<video_engine::VideoFileSource>(config.source);  // Treats source as a video file path.
}  // End of createSource.

std::unique_ptr<video_engine::Pipeline> buildPipeline(
    const AppConfig& config,
    std::shared_ptr<video_engine::IPoseAnalyzer>& pose_analyzer) {  // Builds the processing pipeline.
    auto pipeline = std::make_unique<video_engine::Pipeline>();  // Creates an empty processor chain.
    pipeline->addProcessor(std::make_unique<video_engine::ResizeProcessor>(config.width, config.height));  // Normalizes input frame size first.
    if (config.pipeline == "pose") {  // Builds the pose-estimation pipeline.
        pipeline->addProcessor(std::make_unique<video_engine::PoseEstimator>(  // Adds the pose inference stage.
            config.pose_model, config.pose_config, config.pose_input_width, config.pose_input_height,  // Passes model paths and input dimensions.
            config.pose_confidence, config.pose_backend, config.pose_target));  // Passes confidence and inference device settings.
        if (config.exercise == "squat") {
            video_engine::SquatAnalyzerConfig analyzer_config;
            analyzer_config.minimum_joint_confidence = config.pose_confidence;
            analyzer_config.standing_angle_degrees = config.squat_standing_angle;
            analyzer_config.descent_start_angle_degrees =
                config.squat_descent_start_angle;
            analyzer_config.bottom_angle_degrees = config.squat_bottom_angle;
            analyzer_config.bottom_exit_angle_degrees = config.squat_bottom_exit_angle;
            analyzer_config.minimum_normalized_speed_per_second =
                config.squat_minimum_speed;
            analyzer_config.smoothing_alpha = config.squat_smoothing_alpha;
            pose_analyzer =
                std::make_shared<video_engine::SquatAnalyzer>(analyzer_config);
            pipeline->addProcessor(
                std::make_unique<video_engine::PoseAnalyticsProcessor>(pose_analyzer));
        }
        pipeline->addProcessor(std::make_unique<video_engine::SkeletonRenderer>());  // Adds skeleton drawing after inference.
        if (config.exercise == "squat") {
            pipeline->addProcessor(
                std::make_unique<video_engine::PoseAnalyticsRenderer>());
        }
        return pipeline;  // Returns early because pose mode does not use motion boxes.
    }  // End of pose pipeline branch.

    pipeline->addProcessor(std::make_unique<video_engine::MotionDetector>(  // Adds the motion detection stage.
        config.threshold, config.min_area, config.merge_padding, config.max_detections));  // Passes frame-diff and box merge settings.
    if (config.pipeline == "tracking") {  // Adds tracking only for tracking mode.
        pipeline->addProcessor(std::make_unique<video_engine::ObjectTracker>(config.max_distance, config.max_age, config.trail_length));  // Tracks detections and maintains trails.
    }  // End of tracking branch.
    pipeline->addProcessor(std::make_unique<video_engine::OverlayRenderer>());  // Draws boxes, IDs, and trails.
    return pipeline;  // Returns the motion or tracking pipeline.
}  // End of buildPipeline.

std::string nextOutputPath(const std::string& output_prefix) {  // Finds the first available numbered MP4 path.
    std::filesystem::path prefix(output_prefix);  // Treats the config value as a path prefix.
    std::filesystem::path directory = prefix.parent_path();  // Extracts the output directory.
    if (!directory.empty()) {  // Creates the directory only when the prefix includes one.
        std::filesystem::create_directories(directory);  // Ensures the output folder exists.
    }  // End of output directory setup.

    int index = 0;  // Starts numbering at output0.mp4.
    while (true) {  // Keeps trying names until an unused path is found.
        std::filesystem::path candidate = prefix.string() + std::to_string(index) + ".mp4";  // Builds outputN.mp4.
        if (!std::filesystem::exists(candidate)) {  // Uses this path only if it does not already exist.
            return candidate.string();  // Returns the first available output path.
        }  // End of availability check.
        ++index;  // Tries the next numbered filename.
    }  // End of output path search loop.
}  // End of nextOutputPath.

}  // namespace

int main(int argc, char** argv) {  // Program entry point.
    std::signal(SIGINT, requestStop);  // Lets Ctrl+C leave the loop and finalize video/JSON outputs.
    AppConfig config;  // Holds the final runtime configuration.
    try {
        config = parseArgs(argc, argv);  // Builds the runtime config from CLI and file input.
    } catch (const std::exception& error) {
        std::cerr << "Invalid configuration: " << error.what() << std::endl;  // Reports malformed values or unsupported profiles.
        return 1;  // Stops before opening video devices with an invalid configuration.
    }
    auto source = createSource(config);  // Creates a webcam or video-file source.
    if (!source->open()) {  // Opens the source and checks for failure.
        std::cerr << "Failed to open source: " << config.source << std::endl;  // Reports the failed source path/name.
        return 1;  // Returns non-zero to indicate startup failure.
    }  // End of source open check.

    cv::VideoWriter writer;  // Creates a video writer that starts closed.
    std::string output_video_path;
    if (config.save_output) {  // Enables output writing only when requested.
        output_video_path = nextOutputPath(config.save_path);  // Chooses output0/output1/... without overwriting.
    }  // End of writer setup.

    std::unique_ptr<video_engine::Pipeline> pipeline;  // Owns the configured frame-processing pipeline.
    std::shared_ptr<video_engine::IPoseAnalyzer> pose_analyzer;
    try {
        pipeline = buildPipeline(config, pose_analyzer);  // Creates the pipeline and validates its inference device.
    } catch (const std::exception& error) {
        std::cerr << "Failed to build pipeline: " << error.what() << std::endl;  // Reports model or backend setup failures cleanly.
        return 1;  // Stops before entering the frame loop with an invalid pipeline.
    }
    video_engine::FrameContext ctx;  // Reused per-frame data container passed through the pipeline.
    video_engine::FrameTimeline timeline;  // Owns the monotonic frame and source-time contract.

    size_t frame_index = 0;  // Counts processed frames for FPS calculation.
    auto start_time = video_engine::Profiler::now();  // Captures the start time for average FPS.
    while (!stop_requested) {  // Main frame loop; Ctrl+C requests a graceful stop.
        cv::Mat frame;  // Holds the raw frame read from the input source.
        if (!source->read(frame)) {  // Reads the next frame from the selected source.
            break;  // Stops when the source has no more frames or cannot provide one.
        }  // End of frame read check.
        try {
            timeline.beginFrame(ctx, source->timestampSeconds());  // Starts a fresh pose measurement for this source frame.
        } catch (const std::exception& error) {
            std::cerr << "Invalid frame timeline: " << error.what() << std::endl;
            return 1;
        }
        ctx.raw_frame = frame;  // Stores the raw frame for downstream processors.
        ctx.processed_frame.release();  // Clears the previous processed frame.
        ctx.detections.clear();  // Clears detections from the previous frame.
        ctx.motion_points.clear();  // Clears motion points from the previous frame.
        ctx.stage_names.clear();  // Clears previous profiling stage names.
        ctx.stage_latencies_ms.clear();  // Clears previous profiling stage timings.
        ctx.trails.clear();  // Clears rendered trails; tracker state remains inside ObjectTracker.

        auto stage_start = video_engine::Profiler::now();  // Records pipeline start time.
        pipeline->run(ctx);  // Runs all processors in order; this is the core frame-processing call.
        auto stage_end = video_engine::Profiler::now();  // Records pipeline end time.
        ctx.stage_names.push_back("pipeline");  // Adds a single aggregate pipeline stage label.
        ctx.stage_latencies_ms.push_back(video_engine::Profiler::elapsedMs(stage_start, stage_end));  // Stores total pipeline latency.

        if (ctx.processed_frame.empty()) {  // Ensures there is always a frame to show or write.
            ctx.processed_frame = frame.clone();  // Falls back to the original frame.
        }  // End of processed-frame fallback.

        bool escape_pressed = false;
        if (config.display) {  // Shows the processed frame when display is enabled.
            cv::imshow("Video Engine", ctx.processed_frame);  // Updates the OpenCV display window.
            if (cv::waitKey(1) == 27) {  // Pumps window events and checks for ESC.
                escape_pressed = true;  // Finishes writing this frame before exiting.
            }  // End of key handling.
        }  // End of display branch.

        if (config.save_output && !writer.isOpened()) {
            writer.open(output_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        source->fps(), ctx.processed_frame.size());
            if (!writer.isOpened()) {
                std::cerr << "Failed to open output video: " << output_video_path << std::endl;
                return 1;
            }
            std::cout << "Saving output video to: " << output_video_path << std::endl;
        }
        if (writer.isOpened()) {  // Writes output only when the writer opened successfully.
            writer.write(ctx.processed_frame);  // Appends the processed frame to the output video.
        }  // End of output writing branch.

        ++frame_index;  // Advances the processed frame count.
        auto now = video_engine::Profiler::now();  // Gets the current timestamp.
        double elapsed = std::chrono::duration<double>(now - start_time).count();  // Computes elapsed seconds.
        double fps = elapsed > 0 ? frame_index / elapsed : 0.0;  // Computes average FPS since startup.
        video_engine::Profiler::logFrameStats(frame_index, fps, ctx.stage_names, ctx.stage_latencies_ms);  // Logs frame stats and latency.
        if (escape_pressed) {
            break;
        }
    }  // End of main frame loop.

    writer.release();  // Finalizes the MP4 before reporting completion or writing the summary.
    cv::destroyAllWindows();  // Closes any OpenCV windows.

    if (pose_analyzer) {
        video_engine::SquatSessionSummary summary;
        summary.source = config.source;
        summary.processed_frames = frame_index;
        summary.valid_analysis_frames = pose_analyzer->validFrameCount();
        summary.invalid_analysis_frames = pose_analyzer->invalidFrameCount();
        summary.reps = pose_analyzer->completedReps();
        const auto summary_path = video_engine::SquatSummaryWriter::outputPathForSource(
            config.source, config.analysis_output_dir);
        try {
            video_engine::SquatSummaryWriter::write(summary_path, summary);
        } catch (const std::exception& error) {
            std::cerr << error.what() << std::endl;
            return 1;
        }
        std::cout << "Squat summary saved to: " << summary_path << std::endl;
    }
    return 0;  // Returns success.
}  // End of main.
