# 编译代码
mkdir -p build
cd build
cmake ..
make -j

# 性能测试
./yolo_infer ../../yolov8n.engine ../../image.jpg

# 测试结果
[Detector] engine loaded: ../../yolov8n.engine

[BENCH] baseline (CPU preprocess)
  mean:   4.22 ms
  median: 4.18 ms
  min:    4.01 ms
  max:    5.88 ms
  p99:    5.62 ms
  FPS:    236.7
  [person] conf=0.89
  [person] conf=0.88
  [person] conf=0.88
  [bus] conf=0.84
  [person] conf=0.44