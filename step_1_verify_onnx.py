import onnxruntime as rt
import numpy as np
import cv2

sess = rt.InferenceSession(
    "yolov8n.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# 打印输入输出 shape，后面 c++ 要用
for inp in sess.get_inputs():
    print(f"input: {inp.name}, {inp.shape}, {inp.type}")
for out in sess.get_outputs():
    print(f"output: {out.name}, {out.shape}, {out.type}")

# 随机输入跑一下
dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
result = sess.run(None, {sess.get_inputs()[0].name: dummy})
print(f"output shape: {result[0].shape}")
