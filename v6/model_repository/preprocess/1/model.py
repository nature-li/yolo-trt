import numpy as np
import cv2
import triton_python_backend_utils as pb_utils


INPUT_H = 640
INPUT_W = 640


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            # 取原图字节流
            raw = pb_utils.get_input_tensor_by_name(request, "raw_image")
            buf = raw.as_numpy().tobytes()

            # imdecode
            arr = np.frombuffer(buf, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR, HWC
            src_h, src_w = img.shape[:2]

            # letterbox 参数
            scale = min(INPUT_W / src_w, INPUT_H / src_h)
            new_w = min(int(src_w * scale), INPUT_W)
            new_h = min(int(src_h * scale), INPUT_H)
            pad_x = (INPUT_W - new_w) / 2.0
            pad_y = (INPUT_H - new_h) / 2.0

            # resize
            resized = cv2.resize(img, (new_w, new_h))

            # letterbox canvas，填充 114
            canvas = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8)
            px, py = int(pad_x), int(pad_y)
            canvas[py:py + new_h, px:px + new_w] = resized

            # BGR→RGB, HWC→CHW, /255, float32
            canvas = canvas[:, :, ::-1].astype(np.float32) / 255.0
            chw = np.transpose(canvas, (2, 0, 1))           # [3, 640, 640]
            nchw = chw[np.newaxis, ...]                      # [1, 3, 640, 640]

            t_input  = pb_utils.Tensor("input_tensor", nchw)
            t_scale  = pb_utils.Tensor("scale",  np.array([scale], dtype=np.float32))
            t_pad_x  = pb_utils.Tensor("pad_x",  np.array([pad_x], dtype=np.float32))
            t_pad_y  = pb_utils.Tensor("pad_y",  np.array([pad_y], dtype=np.float32))

            responses.append(pb_utils.InferenceResponse(
                output_tensors=[t_input, t_scale, t_pad_x, t_pad_y]
            ))

        return responses

    def finalize(self):
        pass
