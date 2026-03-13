# 第一步: 导出 onnx
python step_0_export_model.py

# 第二步: 验证 onnx
python step_1_verify_model.py

# 第三步: 转成 egnine
bash step_2_onnx_to_engine.sh

# 第四步： 进入指定版本
cd v1

# 第五步: 编译
mkdir -p build
cd build
cmake ..
make -j

# 第六步： 性能测试
./yolo_infer ../../yolov8n.engine ../../image.jpg
