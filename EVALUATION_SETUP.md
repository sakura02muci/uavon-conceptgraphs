# UAV-ON 离屏评测启动说明

本仓库包含 UAV-ON 的评测实现；为控制仓库大小，不包含数据集、UE4/AirSim 场景包、模型权重、虚拟环境或评测结果。

## 运行模式

标准流程使用 UE4 的 NVIDIA Vulkan 离屏渲染：

```text
-RenderOffscreen -NoSound -NoSplash -NoVSync
```

它不依赖 `Xvfb`、`DISPLAY` 或 SSH 图形转发。UE4 启动后会在本机提供 AirSim RPC，默认端口为 `41451`；评测程序先确认 RPC 可用，再执行 episode。

## 前置条件

- Linux x86_64，NVIDIA GPU 与可用的 NVIDIA 驱动；
- NVIDIA Vulkan ICD 能创建图形设备（CUDA 可用本身不足以保证这一点）；
- Python 3.8 环境，以及 `requirements.txt` 中的 Python 依赖；
- UAV-ON 验证集 JSON，放在仓库同级的 `UAV-ON-dataset/valset/`；
- UAV-ON 测试场景包，放在仓库同级的 `TEST_ENVS/`，例如 `TEST_ENVS/CityPark/CityPark.sh`；
- GroundingDINO/CLIP 所需权重。首次下载权重需要网络；离线机器应预置缓存或权重文件。

期望目录：

```text
workspace/
├── uavon-conceptgraphs/        # 本仓库
├── UAV-ON-dataset/valset/
└── TEST_ENVS/CityPark/
```

## Python 环境

```bash
cd uavon-conceptgraphs
conda create -n uavon python=3.8 -y
conda activate uavon
pip install -r requirements.txt

python - <<'PY'
import airsim, torch, numpy, cv2
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
PY
```

如果项目策略使用 CLIP 或本地 GroundingDINO，请同时确认相应模块和模型权重已按项目配置安装、可加载。

## GPU/Vulkan 检查

```bash
nvidia-smi -L
```

若系统安装了 `vulkan-tools`，还应检查：

```bash
vulkaninfo --summary
```

输出应枚举 NVIDIA GPU，而不是 `llvmpipe`。若 UE4 在日志中加载 Mesa `libvulkan_lvp.so`、出现 `Found no drivers` 或在 RHI 线程崩溃，需要管理员/镜像维护者修复 NVIDIA Vulkan runtime；不要用 Xvfb 替代此问题。

## 推荐：CityPark smoke test

从仓库根目录运行。选择一张当前空闲 GPU；示例使用 CUDA GPU 2：

```bash
export UAVON_GPU_INDEX=2
export UAVON_GENERALIZATION_SCENES=CityPark
export UAVON_GENERALIZATION_SAMPLES_PER_SCENE=1
export UAVON_GENERALIZATION_MAX_STEPS=80
export UAVON_GENERALIZATION_TAG=citypark_smoke_$(date +%Y%m%d_%H%M%S)

bash scripts/run_cross_scene_generalization.sh
```

脚本会：

1. 检查 GPU 空闲显存和利用率；
2. 启动 CityPark，传入 `-RenderOffscreen -NoSound -NoSplash -NoVSync`；
3. 等待 `127.0.0.1:41451` 的 AirSim `confirmConnection()` 成功；
4. 执行一个固定种子的 CityPark episode；
5. 保存结果、UE4 日志和评测日志，并清理本次 UE 进程。

结果在：

```text
results/uavon/<UAVON_GENERALIZATION_TAG>/
├── CityPark.json
├── summary.json
└── logs/
    ├── CityPark.ue4.log
    └── CityPark.eval.log
```

查看结果：

```bash
result_dir="results/uavon/$UAVON_GENERALIZATION_TAG"
cat "$result_dir/summary.json" | python -m json.tool
tail -n 120 "$result_dir/logs/CityPark.ue4.log"
tail -n 120 "$result_dir/logs/CityPark.eval.log"
```

## 批量评测

在 smoke test 稳定后，取消场景筛选即可运行 14 个验证场景：

```bash
unset UAVON_GENERALIZATION_SCENES
export UAVON_GENERALIZATION_SAMPLES_PER_SCENE=1
export UAVON_GENERALIZATION_TAG=generalization_$(date +%Y%m%d_%H%M%S)
bash scripts/run_cross_scene_generalization.sh
```

首次建议每场景只跑一个 episode。确认 GPU、渲染和模型稳定后，再增加 `UAVON_GENERALIZATION_SAMPLES_PER_SCENE`。

## 常见故障

| 现象 | 处理 |
| --- | --- |
| AirSim RPC 未就绪 | 查看 `logs/<scene>.ue4.log`，确认 UE4 没有提前退出。单纯端口可连接不等同于 RPC 可用。 |
| Vulkan/RHI 崩溃 | 检查 NVIDIA Vulkan ICD 与驱动/容器 graphics capability；不要安装或依赖 Xvfb。 |
| GPU 预检拒绝 | 换空闲 GPU，或在允许等待共享 GPU 时设置 `UAVON_GPU_WAIT_SECONDS`。 |
| 图像过黑 | 保留评测中的 render warm-up 与 render-quality 参数。 |
| `Ping returned false` | 检查 `msgpack` 与 `msgpack-rpc-python` 的版本兼容性。 |

