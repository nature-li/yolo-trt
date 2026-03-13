import json
import numpy as np
import triton_python_backend_utils as pb_utils

CONF_THRESH = 0.25
IOU_THRESH  = 0.45
NUM_CLS     = 80
INPUT_W     = 640
INPUT_H     = 640

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def nms(boxes, scores, iou_thresh):
    """boxes: [N,4] xywh, scores: [N]，返回保留的索引"""
    if len(boxes) == 0:
        return []

    # xywh → xyxy
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]

    return keep


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            # 取输入
            raw_out = pb_utils.get_input_tensor_by_name(request, "output0").as_numpy()
            scale   = pb_utils.get_input_tensor_by_name(request, "scale").as_numpy()[0]
            pad_x   = pb_utils.get_input_tensor_by_name(request, "pad_x").as_numpy()[0]
            pad_y   = pb_utils.get_input_tensor_by_name(request, "pad_y").as_numpy()[0]

            # raw_out: [1, 84, 8400] → [84, 8400]
            out = raw_out[0]  # [84, 8400]

            # cx, cy, w, h: [8400]
            cx, cy, bw, bh = out[0], out[1], out[2], out[3]
            # class scores: [80, 8400]
            cls_scores = out[4:]

            # 每个 box 取最高类别分
            class_ids  = cls_scores.argmax(axis=0)        # [8400]
            confidences = cls_scores.max(axis=0)           # [8400]

            # 过滤低置信度
            mask = confidences > CONF_THRESH
            if mask.sum() == 0:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor(
                        "detections",
                        np.array(["[]"], dtype=object)
                    )]
                ))
                continue

            cx  = cx[mask];  cy  = cy[mask]
            bw  = bw[mask];  bh  = bh[mask]
            confidences = confidences[mask]
            class_ids   = class_ids[mask]

            # 坐标还原：input space → normalized [0,1]
            active_w = INPUT_W - 2 * pad_x
            active_h = INPUT_H - 2 * pad_y
            nx = (cx - pad_x) / active_w
            ny = (cy - pad_y) / active_h
            nw = bw / active_w
            nh = bh / active_h

            boxes = np.stack([nx, ny, nw, nh], axis=1)  # [N, 4]

            # 按类别分别做 NMS
            dets = []
            for cls_id in np.unique(class_ids):
                idx = np.where(class_ids == cls_id)[0]
                keep = nms(boxes[idx], confidences[idx], IOU_THRESH)
                for k in keep:
                    i = idx[k]
                    dets.append({
                        "class":    COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else "unknown",
                        "class_id": int(cls_id),
                        "conf":     float(confidences[i]),
                        "x":        float(boxes[i, 0]),
                        "y":        float(boxes[i, 1]),
                        "w":        float(boxes[i, 2]),
                        "h":        float(boxes[i, 3]),
                    })

            # 按置信度排序
            dets.sort(key=lambda d: d["conf"], reverse=True)

            result = np.array([json.dumps(dets)], dtype=object)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("detections", result)]
            ))

        return responses

    def finalize(self):
        pass
