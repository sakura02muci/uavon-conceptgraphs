# ConceptGraphs UAV-ON 配置指南

## 设置 API Key

### 方法 1: 使用 .env 文件（推荐）

1. 复制模板文件：
```bash
cd /villa/mwq24-srt/UAV/UAV_ON
cp .env.template .env
```

2. 编辑 `.env` 文件，填入你的 API key：
```bash
nano .env
# 或
vim .env
```

修改这一行：
```
DEEPSEEK_API_KEY=your-deepseek-api-key-here
```
改为：
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

3. 保存后，所有脚本会自动读取这个配置，无需每次输入。

### 方法 2: 环境变量

```bash
# 临时（当前终端会话）
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx

# 永久（添加到 ~/.bashrc）
echo 'export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx' >> ~/.bashrc
source ~/.bashrc
```

### 方法 3: 命令行参数

```bash
python script.py --api_key sk-xxxxxxxxxxxxxxxx
```

---

## 测试配置

配置完成后，运行测试：

```bash
cd /villa/mwq24-srt/UAV/UAV_ON

# 测试 DeepSeek 连接
PYTHONPATH=src uavon_env/bin/python src/conceptgraphs_uav/deepseek_planner.py \
  --scene_graph scene_graph_clip.json \
  --target vehicle
```

如果看到 `✅ DeepSeek planner initialized` 就说明配置成功了！

---

## 安全提示

⚠️ **重要**: `.env` 文件包含敏感信息，请不要提交到 git

`.gitignore` 已自动排除 `.env` 文件。
