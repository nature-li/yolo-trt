from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.export(format="onnx", imgsz=640, opset=17, simplify=True)
