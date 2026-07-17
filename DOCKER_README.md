# UAV-ON Docker 部署指南

## 前置要求

1. 安装 Docker
2. 安装 NVIDIA Docker Runtime（用于 GPU 支持）
3. 确保 GPU 驱动已正确安装

## 验证 GPU 支持

```bash
# 检查 Docker 是否支持 GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 构建 Docker 镜像

### 方法一：使用构建脚本

```bash
cd /villa/mwq24-srt/ACMMM25-UAV_ON
chmod +x build_docker.sh run_docker.sh
./build_docker.sh
```

### 方法二：手动构建

```bash
cd /villa/mwq24-srt/ACMMM25-UAV_ON
docker build -t uavon:latest .
```

## 运行容器

### 方法一：使用运行脚本

```bash
./run_docker.sh
```

### 方法二：使用 docker-compose

```bash
docker-compose up -d
docker-compose exec uavon bash
```

### 方法三：手动运行

```bash
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -v $(pwd)/DATASET:/workspace/DATASET \
    -v $(pwd)/TRAIN_ENVS:/workspace/TRAIN_ENVS \
    -v $(pwd)/TEST_ENVS:/workspace/TEST_ENVS \
    --shm-size=16g \
    --network host \
    uavon:latest bash
```

## 容器内使用

进入容器后：

```bash
# 验证 GPU 可用性
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# 启动 AirSim 服务器
python airsim_plugin/AirVLNSimulatorServerTool.py --port=30000 --root_path=/workspace

# 运行评估（在另一个终端）
bash scripts/eval_fixed.sh
```

## 目录结构

项目应该按以下结构组织：

```
/villa/mwq24-srt/ACMMM25-UAV_ON/
├── Dockerfile
├── docker-compose.yml
├── build_docker.sh
├── run_docker.sh
├── requirements.txt
├── DATASET/          # 需要下载
├── TRAIN_ENVS/       # 需要下载 (44.1G)
└── TEST_ENVS/        # 需要下载 (26.8G)
```

## 下载数据

### 环境数据

- 训练环境: https://huggingface.co/datasets/Kyaren/UAV-ON-envs-train (44.1G)
- 测试环境: https://huggingface.co/datasets/Kyaren/UAV-ON-envs-test (26.8G)

### 数据集

- 数据集: https://huggingface.co/datasets/Kyaren/UAV-ON-dataset

下载后解压到对应目录。

## 常用命令

```bash
# 查看运行中的容器
docker ps

# 进入运行中的容器
docker exec -it uavon_container bash

# 停止容器 (docker-compose)
docker-compose down

# 查看容器日志
docker logs uavon_container

# 重新构建镜像
docker build --no-cache -t uavon:latest .
```

## 故障排除

### GPU 不可用

如果遇到 "Ping returned false" 错误：

```bash
pip uninstall msgpack-python msgpack-rpc-python
pip install msgpack-rpc-python
```

### 权限问题

如果遇到文件权限问题，可以在容器内运行：

```bash
chown -R $(id -u):$(id -g) /workspace
```
