FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 使用阿里云镜像源
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /workspace

COPY requirements.txt .

# 使用清华源加速pip安装
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --no-cache-dir numpy==1.24.4 && \
    pip3 install --no-cache-dir msgpack-python==0.5.6 msgpack-rpc-python==0.4.1 && \
    pip3 install --no-cache-dir --no-build-isolation airsim==1.8.1 && \
    pip3 install --no-cache-dir -r requirements.txt || true

COPY . .

CMD ["/bin/bash"]
