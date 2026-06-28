# Real-Time Pose and Exercise Analytics Engine

A modular C++17/OpenCV engine for real-time video processing, human-pose inference, and
exercise analytics on macOS and NVIDIA Jetson Orin Nano.

The project turns webcam or recorded video into motion regions, tracked trajectories, or
an 18-joint OpenPose skeleton. Its squat mode derives joint angles, movement speed,
phases, and repetitions from reliable per-frame timestamps while preserving structured
results for future subject tracking and learned analytics.

> Current state: the video engine, pose pipeline, canonical pose schema, timing contract,
> mathematical squat analytics, platform profiles, and tests are implemented. Persistent
> primary-person tracking and a faster TensorRT pose model are on the roadmap.

## What Has Been Built

- Webcam and recorded-video input
- Motion detection with merged foreground regions
- Centroid-based object tracking and trajectory trails
- OpenPose COCO 18-joint inference through OpenCV DNN
- Fresh pose inference for every decoded frame, with no stale-result reuse
- Canonical named-joint pose data independent of the current model
- Monotonic frame IDs and source-relative timestamps for motion calculations
- Optional depth and 3D fields reserved in the pose interface
- Per-stage latency for preprocessing, inference, postprocessing, rendering, and the
  complete processor pipeline
- Real-time squat phase, joint-angle, speed, and repetition analysis in a side panel
- Input-named JSON squat summaries with per-rep metrics
- CPU configuration for macOS and CUDA FP16 configuration for Jetson Orin Nano
- Automated tests for motion, tracking, rendering, timeline behavior, pose mapping, and
  squat state semantics

The current Caffe model produces 2D keypoints. Those results are available as structured
data for the squat analyzer and future learned analytics instead of being limited to the
skeleton overlay.

## Demo

The executable exposes three runtime pipelines:

| Pipeline | Input | Result |
| --- | --- | --- |
| `motion` | Webcam or video | Foreground regions and bounding boxes |
| `tracking` | Webcam or video | Stable motion-track IDs and trajectory trails |
| `pose` | Webcam or video | 18-joint skeleton, confidence, and stage timing |

The pose demo can display a live OpenCV window or save an annotated video. A typical
runtime report looks like:

```text
[Frame 3] FPS: 3.4
| resize: 0.40 ms
| pose_preprocess: 0.41 ms
| pose_inference: 261.56 ms
| pose_postprocess: 0.01 ms
| pose_estimator: 261.98 ms
| skeleton_renderer: 0.10 ms
| pipeline: 262.47 ms
```

Run the included squat video:

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --config configs/squat.yaml
```

The live result places the video on the left and squat phase, angles, speed, and rep count
on the right. The final summary is written to `output/squat.json`. Use
`--save output/pose_demo` to write an annotated video.

## Architecture

```text
Webcam / Video File
        |
        v
   VideoSource
        |
        v
  FrameTimeline
  frame ID + source time + fresh validity state
        |
        v
 ResizeProcessor
        |
        +----------------------------+
        |                            |
        v                            v
 MotionDetector                PoseEstimator
        |                   preprocess -> DNN -> decode
        v                            |
 ObjectTracker                      v
        |                    Canonical Pose Schema
        v                            |
 OverlayRenderer          PoseAnalyticsProcessor
        |                            |
        |                            v
        |                    SkeletonRenderer
        |                            |
        |                            v
        |                 PoseAnalyticsRenderer
        |                            |
        +-------------+--------------+
                      v
             Display / Video Output
```

All frame processors implement a shared `IFrameProcessor` interface and exchange data
through `FrameContext`. Pose-model channel indices are translated by
`OpenPoseCocoAdapter` into named joints before rendering or analytics. This keeps future
ONNX/TensorRT models and exercise-analysis modules separate from the current Caffe
implementation.

The CPU currently handles capture, resize, orchestration, pose decoding, analytics-ready
data, rendering, and output. Neural-network inference runs on the configured OpenCV DNN
target:

- macOS profile: CPU
- Jetson Orin Nano profile: CUDA FP16
- Planned Orin production path: lightweight ONNX model with TensorRT FP16

For the complete frame lifecycle and CPU/GPU boundary, see the
[Video Engine Guide](docs/VIDEO_ENGINE_GUIDE.md) and
[Pose Inference Guide](docs/POSE_INFERENCE_GUIDE.md).

## How to Run

### Requirements

- CMake 3.16 or newer
- A C++17 compiler
- OpenCV with `core`, `imgproc`, `highgui`, `videoio`, and `dnn`
- OpenPose COCO model files:
  - `models/pose_deploy_linevec.prototxt`
  - `models/pose_iter_440000.caffemodel`

The repository includes the model definition but not the large Caffe weights. Place the
weights at the path above before running pose inference.

### Build and test

Use a Release build for meaningful performance measurements:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

### macOS

Run pose inference with the CPU profile:

```bash
./build/video_engine \
  --pipeline pose \
  --config configs/pose.yaml
```

### Jetson Orin Nano

Build the project directly on the Jetson using an OpenCV installation compiled with
CUDA, cuDNN, and OpenCV DNN CUDA support. Then run:

```bash
./build/video_engine \
  --pipeline pose \
  --config configs/pose_jetson.yaml
```

The macOS and Jetson binaries are built separately. For JetPack/OpenCV requirements,
headless execution, display forwarding, and troubleshooting, use the
[Platform Run Guide](docs/PLATFORM_RUN_GUIDE.md).

### Other pipelines

```bash
# Motion detection from the webcam
./build/video_engine \
  --source webcam \
  --pipeline motion \
  --config configs/motion.yaml

# Motion tracking on the included video
./build/video_engine \
  --source video_source/squat.mov \
  --pipeline tracking \
  --config configs/tracking.yaml
```

Common overrides:

| Option | Purpose |
| --- | --- |
| `--source webcam` | Use the default camera |
| `--source <path>` | Read a video file |
| `--pipeline motion\|tracking\|pose` | Select the processor pipeline |
| `--inference-platform cpu` | Force OpenCV DNN CPU inference |
| `--inference-platform jetson` | Select CUDA FP16 inference |
| `--exercise squat` | Enable mathematical squat analysis |
| `--analysis-output <dir>` | Select the JSON summary directory |
| `--display` / `--no-display` | Enable or disable the GUI window |
| `--save <prefix>` / `--no-save` | Enable or disable annotated video output |

## Current Benchmark

Representative frame from the completed 117-frame macOS CPU squat validation:

| Stage | Example latency |
| --- | ---: |
| Resize | 0.28 ms |
| Pose preprocessing | 0.26 ms |
| Neural-network inference | 262.30 ms |
| Pose postprocessing | 0.01 ms |
| Complete pose estimator | 262.57 ms |
| Squat analytics | <0.01 ms |
| Skeleton rendering | 0.09 ms |
| Analytics side panel | 0.60 ms |
| Processor pipeline | 263.55 ms |
| Observed throughput | 3.8 FPS |

Inference is the dominant cost; resizing and drawing are currently negligible by
comparison. This points the next optimization effort toward a smaller model and an
accelerated inference backend.

These numbers are a development baseline, not a formal cross-platform benchmark. The
Jetson Orin Nano profile is prepared but has not yet been measured on the physical
device. Its project target is at least **20 FPS** and less than **100 ms** end-to-end
latency.

For a reproducible benchmark procedure and an explanation of every timing field, see the
[Pose Inference Guide](docs/POSE_INFERENCE_GUIDE.md).

## Roadmap

### Next

- Add persistent primary-person tracking
  - retain the selected athlete when other people enter the frame
  - use position, bounding-box overlap, keypoint similarity, and reacquisition rules

### After that

- Replace the heavy Caffe network with a faster, accurate multi-person ONNX pose model
- Use one canonical keypoint contract across macOS and Jetson
- Integrate TensorRT FP16, then evaluate eligible INT8 execution on Orin Nano
- Benchmark power mode, temperature, Tensor Core use, and CPU/GPU transfers
- Evaluate hardware decoding, asynchronous stages, and zero-copy only where profiling
  demonstrates a meaningful gain
- Connect stereo/depth input through the reserved depth and 3D pose interface
- Keep the mathematical analytics API open for future learned exercise-analysis models

Detailed decisions, dependencies, and implementation tickets are maintained in the
[Pose Analytics Decision Map](docs/POSE_ANALYTICS_DECISION_MAP.md).

## More Documentation

| Page | Contents |
| --- | --- |
| [Video Engine Guide](docs/VIDEO_ENGINE_GUIDE.md) | Full runtime architecture, processors, configuration, and extension points |
| [Pose Inference Guide](docs/POSE_INFERENCE_GUIDE.md) | Pose data flow, timing, settings, CPU/GPU boundaries, and benchmarking |
| [Squat Analytics Guide](docs/SQUAT_ANALYTICS_GUIDE.md) | Live metrics, rep semantics, settings, JSON output, and limitations |
| [Platform Run Guide](docs/PLATFORM_RUN_GUIDE.md) | macOS and Jetson Orin Nano setup and commands |
| [Pose Analytics Decision Map](docs/POSE_ANALYTICS_DECISION_MAP.md) | Resolved design choices, roadmap tickets, and acceptance criteria |
| [Resources](RESOURCES.md) | NVIDIA, OpenCV, TensorRT, tracking, and calibration references |

Quick references:

- [Jetson Pipeline Cheat Sheet](reference/jetson-pipeline-cheatsheet.html)
- [Pose Analytics Math Cheat Sheet](reference/pose-analytics-cheatsheet.html)
