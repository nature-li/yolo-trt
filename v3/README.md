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
  mean:   2.28 ms
  median: 2.27 ms
  min:    2.18 ms
  max:    2.40 ms
  p99:    2.35 ms
  FPS:    439.2
  class_id=0 conf=0.88 box=[221,409,343,857] class_name=person
  class_id=0 conf=0.88 box=[671,390,809,877] class_name=person
  class_id=5 conf=0.86 box=[29,229,797,768] class_name=bus
  class_id=0 conf=0.85 box=[51,398,242,905] class_name=person
  class_id=0 conf=0.37 box=[0,547,58,873] class_name=person
saved result.jpg