# docker build -t trt-dev .
FROM nvcr.io/nvidia/tensorrt:25.01-py3

# 防止 apt 交互
ENV DEBIAN_FRONTEND=noninteractive

# 基础开发工具（镜像里大多已有，这里保险）
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
 && rm -rf /var/lib/apt/lists/*

# pip 安装
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir \
    onnxruntime-gpu \
    opencv-python-headless \
    -i https://mirrors.aliyun.com/pypi/simple/

# 工作目录
WORKDIR /workspace

# 默认进容器给 bash
CMD ["/bin/bash"]

