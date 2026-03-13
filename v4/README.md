# 编译代码
mkdir -p build
cd build
cmake ..
make -j

# 性能测试
./yolo_infer ../../yolov8n.engine ../../image.jpg


# 测试结果
[BENCH] baseline (CPU preprocess)
  mean:   1.02 ms
  median: 1.02 ms
  min:    1.01 ms
  max:    1.25 ms
  p99:    1.14 ms
  FPS:    976.2
  class_id=0 conf=0.88 box=[221,409,343,857] class_name=person
  class_id=0 conf=0.88 box=[671,390,809,877] class_name=person
  class_id=5 conf=0.86 box=[29,229,797,768] class_name=bus
  class_id=0 conf=0.85 box=[51,398,242,905] class_name=person
  class_id=0 conf=0.37 box=[0,547,58,873] class_name=person
saved result.jpg