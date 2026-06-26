# Modern C++ OpenCV Video Engine

A compact C++17/OpenCV demo project for motion detection, robust object tracking, trajectory trails, and detailed latency reporting.

For a detailed explanation of the runtime flow, processor pipeline, configuration, debugging, and model replacement points, see [Video Engine Guide](docs/VIDEO_ENGINE_GUIDE.md).

## Build

```bash
cmake -S . -B build
cmake --build build -j4
```

## Run

```bash
./build/video_engine --source webcam --pipeline motion --display
./build/video_engine --source videos/demo.mp4 --pipeline tracking --config configs/tracking.yaml
./build/video_engine --pipeline pose --config configs/pose.yaml
```

## Pose Skeleton

Pose mode draws head, shoulder, elbow, wrist, hip, knee, and ankle keypoints with skeleton lines.

Place an OpenPose COCO model in `models/`:

```text
models/pose_deploy_linevec.prototxt
models/pose_iter_440000.caffemodel
```

Then run:

```bash
./build/video_engine --pipeline pose --config configs/pose.yaml
```

For higher FPS, `configs/pose.yaml` defaults to a smaller pose input and runs model inference
every 3 frames while reusing the latest skeleton on skipped frames.

You can switch OpenCV DNN backend/target in `configs/pose.yaml`, for example:

```yaml
pose_backend: openvino
pose_target: cpu
```

## Tests

```bash
ctest --test-dir build --output-on-failure
```
