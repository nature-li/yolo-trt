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
  mean:   2.87 ms
  median: 2.82 ms
  min:    2.65 ms
  max:    7.53 ms
  p99:    4.70 ms
  FPS:    347.9
  class_id=0 conf=0.89 box=[670,380,809,879] class_name=person
  class_id=0 conf=0.88 box=[221,407,343,856] class_name=person
  class_id=0 conf=0.88 box=[50,397,244,905] class_name=person
  class_id=5 conf=0.84 box=[31,230,801,775] class_name=bus
  class_id=0 conf=0.44 box=[0,549,57,868] class_name=person
saved result.jpg