# Modern C++ OpenCV Video Engine

A compact C++17/OpenCV demo project for motion detection, robust object tracking, trajectory trails, and detailed latency reporting.

## Build

```bash
cmake -S . -B build
cmake --build build -j4
```

## Run

```bash
./build/video_engine --source webcam --pipeline motion --display
./build/video_engine --source videos/demo.mp4 --pipeline tracking --config configs/tracking.yaml
```

## Tests

```bash
ctest --test-dir build --output-on-failure
```
