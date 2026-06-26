#pragma once

#include <algorithm>  // Normalizes backend/target strings and replaces separators.
#include <cctype>     // Converts backend/target names to lowercase safely.
#include <stdexcept>  // Reports model loading and backend configuration errors.
#include <string>     // Stores model paths and backend/target names.
#include <vector>     // Stores cached pose results.

#include <opencv2/core.hpp>    // Provides cv::Mat, cv::Point, and scalar types.
#include <opencv2/dnn.hpp>     // Provides OpenCV DNN model loading and inference.
#include <opencv2/imgproc.hpp> // Provides blob/image preprocessing helpers.

#include "core/IFrameProcessor.hpp"  // Defines the frame processor interface.

namespace video_engine {

class PoseEstimator : public IFrameProcessor {
public:
    explicit PoseEstimator(std::string model_weights, std::string model_config = "",
                           int input_width = 368, int input_height = 368,
                           float confidence_threshold = 0.12F, int inference_interval = 1,
                           std::string backend = "opencv", std::string target = "cpu")
        : input_width_(input_width),
          input_height_(input_height),
          confidence_threshold_(confidence_threshold),
          inference_interval_(inference_interval < 1 ? 1 : inference_interval) {
        if (model_weights.empty()) {
            throw std::runtime_error("Pose model path is empty. Set pose_model in configs/pose.yaml.");
        }

        net_ = model_config.empty() ? cv::dnn::readNet(model_weights)
                                    : cv::dnn::readNet(model_weights, model_config);
        if (net_.empty()) {
            throw std::runtime_error("Failed to load pose model: " + model_weights);
        }

        net_.setPreferableBackend(parseBackend(backend));
        net_.setPreferableTarget(parseTarget(target));
    }

    void process(FrameContext& ctx) override {
        if (ctx.processed_frame.empty()) {
            return;
        }

        const bool should_run_inference =
            cached_poses_.empty() || inference_interval_ <= 1 ||
            frame_counter_ % static_cast<size_t>(inference_interval_) == 0;
        ++frame_counter_;

        if (!should_run_inference) {
            ctx.poses = cached_poses_;
            return;
        }

        cv::Mat blob = cv::dnn::blobFromImage(ctx.processed_frame, 1.0 / 255.0,
                                              cv::Size(input_width_, input_height_),
                                              cv::Scalar(0, 0, 0), false, false);
        net_.setInput(blob);
        cv::Mat output = net_.forward();

        if (output.dims != 4 || output.size[1] < kKeypointCount) {
            throw std::runtime_error("Pose model output is not OpenPose COCO 18-keypoint format.");
        }

        const int heatmap_height = output.size[2];
        const int heatmap_width = output.size[3];
        const int frame_width = ctx.processed_frame.cols;
        const int frame_height = ctx.processed_frame.rows;

        Pose pose;
        pose.keypoints.resize(kKeypointCount);
        for (int part = 0; part < kKeypointCount; ++part) {
            cv::Mat heatmap(heatmap_height, heatmap_width, CV_32F, output.ptr(0, part));

            double confidence = 0.0;
            cv::Point max_location;
            cv::minMaxLoc(heatmap, nullptr, &confidence, nullptr, &max_location);

            PoseKeypoint keypoint;
            keypoint.confidence = static_cast<float>(confidence);
            keypoint.visible = confidence >= confidence_threshold_;
            keypoint.point = cv::Point(
                static_cast<int>((frame_width * max_location.x) / static_cast<float>(heatmap_width)),
                static_cast<int>((frame_height * max_location.y) / static_cast<float>(heatmap_height)));
            pose.keypoints[part] = keypoint;
        }

        ctx.poses.clear();
        ctx.poses.push_back(pose);
        cached_poses_ = ctx.poses;
    }

    std::string name() const override {
        return "pose_estimator";
    }

    static constexpr int kKeypointCount = 18;

private:
    int parseBackend(std::string backend) const {
        normalize(backend);
        if (backend == "default") {
            return cv::dnn::DNN_BACKEND_DEFAULT;
        }
        if (backend == "opencv") {
            return cv::dnn::DNN_BACKEND_OPENCV;
        }
        if (backend == "openvino" || backend == "inference_engine") {
            return cv::dnn::DNN_BACKEND_INFERENCE_ENGINE;
        }
        if (backend == "cuda") {
            return cv::dnn::DNN_BACKEND_CUDA;
        }
        if (backend == "vkcom" || backend == "vulkan") {
            return cv::dnn::DNN_BACKEND_VKCOM;
        }
        throw std::runtime_error("Unsupported pose backend: " + backend);
    }

    int parseTarget(std::string target) const {
        normalize(target);
        if (target == "cpu") {
            return cv::dnn::DNN_TARGET_CPU;
        }
        if (target == "cpu_fp16" || target == "fp16") {
            return cv::dnn::DNN_TARGET_CPU_FP16;
        }
        if (target == "opencl" || target == "gpu") {
            return cv::dnn::DNN_TARGET_OPENCL;
        }
        if (target == "opencl_fp16" || target == "gpu_fp16") {
            return cv::dnn::DNN_TARGET_OPENCL_FP16;
        }
        if (target == "vulkan") {
            return cv::dnn::DNN_TARGET_VULKAN;
        }
        if (target == "cuda") {
            return cv::dnn::DNN_TARGET_CUDA;
        }
        if (target == "cuda_fp16") {
            return cv::dnn::DNN_TARGET_CUDA_FP16;
        }
        if (target == "myriad" || target == "vpu") {
            return cv::dnn::DNN_TARGET_MYRIAD;
        }
        if (target == "hddl") {
            return cv::dnn::DNN_TARGET_HDDL;
        }
        if (target == "npu") {
            return cv::dnn::DNN_TARGET_NPU;
        }
        throw std::runtime_error("Unsupported pose target: " + target);
    }

    void normalize(std::string& value) const {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        std::replace(value.begin(), value.end(), '-', '_');
    }

    int input_width_;
    int input_height_;
    float confidence_threshold_;
    int inference_interval_;
    size_t frame_counter_ = 0;
    std::vector<Pose> cached_poses_;
    cv::dnn::Net net_;
};

}  // namespace video_engine
