#!/usr/bin/env python3
"""
测试客户端
pip install tritonclient[http] numpy
"""
import sys
import json
import numpy as np
import tritonclient.http as httpclient


def infer(image_path: str, url: str = "localhost:8000"):
    client = httpclient.InferenceServerClient(url=url)

    # 读图片，压成字节流
    with open(image_path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)

    # 输入
    inputs = [httpclient.InferInput("raw_image", buf.shape, "UINT8")]
    inputs[0].set_data_from_numpy(buf)

    # 输出
    outputs = [httpclient.InferRequestedOutput("detections")]

    # 发请求
    resp = client.infer(
        model_name="yolov8n_ensemble",
        inputs=inputs,
        outputs=outputs,
    )

    # 解析结果
    dets_json = resp.as_numpy("detections")[0]
    dets = json.loads(dets_json)

    print(f"Detected {len(dets)} objects:")
    for d in dets:
        print(
            f"  {d['class']:15s} conf={d['conf']:.3f} "
            f"x={d['x']:.3f} y={d['y']:.3f} w={d['w']:.3f} h={d['h']:.3f}"
        )


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    infer(image_path)
