# UAV-ON Dataset Evaluation Guide

## 概述

本文档说明如何使用 UAV-ON 标准数据集评估 ConceptGraphs 导航系统。

## 数据集结构

```
UAV-ON-dataset/
├── trainset/          # 训练集（用于微调模型）
│   ├── Barnyard.json
│   ├── BrushifyRoad.json
│   ├── ...
└── valset/           # 验证集（用于评估）
    ├── Barnyard.json      # 14 episodes
    ├── BrushifyRoad.json  # ...
    ├── BrushifyUrban.json
    ├── CabinLake.json
    ├── CityPark.json
    ├── CityStreet.json
    ├── DownTown.json
    ├── Neighborhood.json
    ├── NYC.json
    ├── Slum.json
    ├── UrbanJapan.json
    ├── Venice.json
    ├── WesternTown.json
    └── WinterTown.json
```

## Episode 数据格式

每个 JSON 文件包含多个 episode：

```json
{
    "episode_id": "0",
    "map_name": "Barnyard_test",
    "true_name": " BusStop",
    "object_name": "BusStop_2",
    "pose": [[x, y, z]],                    # 目标位置
    "start_pose": {
        "start_position": [x, y, z],        # 起始位置
        "start_quaternionr": [x, y, z, w]   # 起始朝向（四元数）
    },
    "info": {
        "geodesic_distance": 72,           # 最短路径距离（米）
        "euclidean_distance": 50.7181      # 直线距离（米）
    },
    "size": " big(5.0*3.0=15.0 squares)",
    "description": "...",
    "category": "",
    "used-in-train": 0,
    "commonsenible": 1
}
```

## 评估指标

UAV-ON 使用以下标准指标：

1. **Success Rate (SR)**: 成功找到目标的比例
   - 成功条件：距离目标 < 5 米
   
2. **Success weighted by Path Length (SPL)**:
   ```
   SPL = (1/N) * Σ S_i * (L_i / max(P_i, L_i))
   ```
   - S_i: episode i 是否成功（0或1）
   - L_i: episode i 的 geodesic distance
   - P_i: episode i 的实际路径长度
   
3. **Distance to Goal (DtG)**: 距离目标的最小距离

## 环境启动

每个场景需要独立启动 UE4 环境：

```bash
# 进入场景目录
cd TEST_ENVS/Barnyard/

# 启动 Barnyard 场景
./Barnyard_test1.sh -windowed -ResX=1024 -ResY=768
```

可用场景：
- Barnyard
- BrushifyRoad
- BrushifyUrban
- CabinLake
- CityPark
- CityStreet (CleanCityStreet)
- DownTown
- Neighborhood
- NYC
- Slum
- UrbanJapan
- Venice
- WesternTown
- WinterTown

## 运行评估

### 方法 1: 简单评估脚本

使用我们创建的简化评估脚本：

```bash
cd UAV_ON

# 评估单个场景的前5个episode
python scripts/eval_simple_uavon.py \
    --dataset ../UAV-ON-dataset/valset/Barnyard.json \
    --num_episodes 5 \
    --strategy clip \
    --output results/barnyard_clip.json

# 使用 DeepSeek 策略
python scripts/eval_simple_uavon.py \
    --dataset ../UAV-ON-dataset/valset/Barnyard.json \
    --num_episodes 5 \
    --strategy deepseek \
    --output results/barnyard_deepseek.json
```

### 方法 2: 完整 UAV-ON 框架评估

使用 UAV-ON 的原生评估框架（需要更多配置）：

```bash
cd UAV_ON

# 使用 ConceptGraphs 作为模型包装器
python src/eval_2.py \
    --dataset_path ../UAV-ON-dataset/valset/Barnyard.json \
    --eval_save_path results/conceptgraphs \
    --name conceptgraphs_clip \
    --batchSize 1 \
    --maxActions 500
```

## 评估流程

1. **启动环境**
   ```bash
   # 终端 1: 启动 UE4 场景
   cd TEST_ENVS/Barnyard
   ./Barnyard_test1.sh -windowed
   ```

2. **运行评估**
   ```bash
   # 终端 2: 运行评估脚本
   cd UAV_ON
   python scripts/eval_simple_uavon.py \
       --dataset ../UAV-ON-dataset/valset/Barnyard.json \
       --strategy clip
   ```

3. **查看结果**
   ```bash
   cat results/barnyard_clip.json | python -m json.tool
   ```

## 批量评估所有场景

```bash
#!/bin/bash
cd UAV_ON

SCENES=(
    "Barnyard"
    "BrushifyRoad"
    "BrushifyUrban"
    "CabinLake"
    "CityPark"
    "CityStreet"
    "DownTown"
    "Neighborhood"
    "NYC"
    "Slum"
    "UrbanJapan"
    "Venice"
    "WesternTown"
    "WinterTown"
)

for scene in "${SCENES[@]}"; do
    echo "Evaluating $scene..."
    
    # 启动环境（需要手动操作）
    # cd ../TEST_ENVS/$scene
    # ./启动脚本.sh &
    # sleep 10
    
    # 运行评估
    python scripts/eval_simple_uavon.py \
        --dataset ../UAV-ON-dataset/valset/${scene}.json \
        --strategy clip \
        --output results/${scene}_clip.json
    
    # 关闭环境
    # killall -9 *-Linux-Shipping
done
```

## 结果分析

评估完成后，结果保存为 JSON 格式：

```json
{
    "summary": {
        "dataset": "Barnyard.json",
        "strategy": "clip",
        "num_episodes": 14,
        "success_rate": 0.857,
        "mean_spl": 0.672,
        "mean_distance_to_goal": 2.34,
        "mean_path_length": 45.2
    },
    "episodes": [
        {
            "episode_id": "0",
            "success": true,
            "spl": 0.753,
            "path_length": 52.3,
            ...
        },
        ...
    ]
}
```

## 与 UAV-ON Baseline 对比

UAV-ON 论文中的 baseline 结果（参考）：

| 方法 | Success Rate | SPL |
|------|-------------|-----|
| Random | ~5% | ~0.02 |
| FMM | ~15% | ~0.08 |
| CLIP-H | ~35% | ~0.22 |
| **ConceptGraphs (我们)** | **待评估** | **待评估** |

## 注意事项

1. **环境要求**
   - 每个场景需要单独的 UE4 环境运行
   - 需要足够的 GPU 内存（推荐 ≥8GB）
   - AirSim 连接端口：默认 41451

2. **评估设置**
   - 最大步数：500
   - 成功阈值：5 米
   - 图像分辨率：256×144（UE4 设置）

3. **已知限制**
   - 低分辨率图像导致视觉模糊
   - CLIP 置信度相对较低（0.03-0.05）
   - 需要调优detection阈值和导航策略

## 下一步

1. 在所有14个场景上运行完整评估
2. 与 UAV-ON baseline 对比
3. 调优超参数（detection阈值、探索策略）
4. 生成完整的评估报告
