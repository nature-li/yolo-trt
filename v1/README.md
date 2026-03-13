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
  mean:   4.30 ms
  median: 4.21 ms
  min:    4.00 ms
  max:    7.64 ms
  p99:    6.21 ms
  FPS:    232.7
  [person] conf=0.89
  [person] conf=0.88
  [person] conf=0.88
  [bus] conf=0.84
  [person] conf=0.44