# YOLOv8n Triton Inference Server

## 目录结构

```
model_repository/
  preprocess/          # Python backend，letterbox
  yolov8n/             # TensorRT backend
  postprocess/         # Python backend，decode + NMS
  yolov8n_ensemble/    # Ensemble，串联上面三个
client.py              # 测试客户端
```

## 部署步骤

### 1. 准备 engine 文件

```bash
ln -s /path/to/yolov8n.engine model_repository/yolov8n/1/model.plan
```

### 2. 启动 Triton

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:25.01-py3 \
  tritonserver --model-repository=/models
```

### 3. 测试

```bash
pip install tritonclient[http] numpy opencv-python
python client.py /path/to/image.jpg
```

### 4. 健康检查

```bash
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/yolov8n_ensemble/ready
```

### 5. metrics（Prometheus 格式）

```bash
curl http://localhost:8002/metrics
```

## 端口说明

| 端口 | 协议 |
|------|------|
| 8000 | HTTP |
| 8001 | gRPC |
| 8002 | metrics |

### 6.压测
bash pressure.sh


### 7.压测结果
done, size: 137419
Running 10s test @ http://localhost:8000
  4 threads and 16 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   140.60ms    8.78ms 163.01ms   90.01%
    Req/Sec    28.29      9.06    40.00     66.17%
  1131 requests in 10.01s, 1.27MB read
Requests/sec:    113.03
Transfer/sec:    130.47KB