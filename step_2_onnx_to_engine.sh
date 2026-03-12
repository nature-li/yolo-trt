trtexec \
    --onnx=yolov8n.onnx  \
    --saveEngine=yolov8n.engine \
    --fp16 \
    --memPoolSize=workspace:4096
