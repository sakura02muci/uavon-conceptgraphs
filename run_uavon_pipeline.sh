#!/bin/bash
# UAV-ON ConceptGraph + ObjectNav Pipeline
# 同一条 UAV-ON episode 轨迹中同步生成场景图和评估指标

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================="
echo "  UAV-ON ConceptGraphs 场景图生成和评估"
echo "======================================================================="
echo ""

# 配置
DATASET="../UAV-ON-dataset/valset/Barnyard.json"
NUM_EPISODES=3
MAX_STEPS=150
STRATEGY="conceptgraph"
RESULTS_DIR="results/uavon"
GRAPH_DIR="$RESULTS_DIR/scene_graphs_${STRATEGY}"
FRAMES_DIR="$RESULTS_DIR/frames_${STRATEGY}"

# 创建结果目录
mkdir -p "$RESULTS_DIR"

echo "📋 配置信息:"
echo "   Dataset: $DATASET"
echo "   Episodes: $NUM_EPISODES"
echo "   Max Steps: $MAX_STEPS"
echo "   Strategy: $STRATEGY"
echo ""

# 步骤 1: 环境测试
echo "═══════════════════════════════════════════════════════════════════════"
echo "步骤 1/3: 测试环境配置"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

python scripts/test_uavon_setup.py || {
    echo "❌ 环境测试失败，请检查配置"
    exit 1
}

echo ""
echo "✅ 环境测试通过"
echo ""
sleep 1

# 步骤 2: 标准评估 + 在线构建场景图
echo "═══════════════════════════════════════════════════════════════════════"
echo "步骤 2/3: UAV-ON ObjectNav 评估并同步构建 ConceptGraph"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

python scripts/eval_simple_uavon.py \
    --dataset "$DATASET" \
    --num-episodes $NUM_EPISODES \
    --strategy $STRATEGY \
    --max-steps $MAX_STEPS \
    --output "$RESULTS_DIR/evaluation_${STRATEGY}.json" \
    --graph-dir "$GRAPH_DIR" \
    --save-frames-dir "$FRAMES_DIR" || {
    echo "⚠️  评估过程中断，但可能有部分结果"
}

echo ""
echo "✅ 评估和场景图生成完成"
echo ""
sleep 1

# 步骤 3: 结果聚合
echo "═══════════════════════════════════════════════════════════════════════"
echo "步骤 3/3: 生成评估报告"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

if [ -f "$RESULTS_DIR/evaluation_${STRATEGY}.json" ]; then
    python scripts/aggregate_uavon_results.py \
        --strategy $STRATEGY \
        --results_dir "$RESULTS_DIR" \
        --output "$RESULTS_DIR/report_${STRATEGY}.txt" || {
        echo "⚠️  报告生成失败"
    }
    
    echo ""
    echo "✅ 评估报告已生成"
    echo ""
    
    if [ -f "$RESULTS_DIR/report_${STRATEGY}.txt" ]; then
        echo "-------------------------------------------------------------------"
        echo "报告摘要:"
        echo "-------------------------------------------------------------------"
        head -40 "$RESULTS_DIR/report_${STRATEGY}.txt"
        echo "..."
        echo "完整报告: $RESULTS_DIR/report_${STRATEGY}.txt"
    fi
else
    echo "⚠️  未找到评估结果文件"
fi

# 总结
echo ""
echo "======================================================================="
echo "  完成!"
echo "======================================================================="
echo ""
echo "生成的文件:"
ls -lh "$RESULTS_DIR"/ | grep -E "\.json$|\.txt$|\.png$"
echo ""
echo "查看结果:"
echo "  场景图: ls $GRAPH_DIR/*.json"
echo "  RGB-D帧: ls $FRAMES_DIR"
echo "  评估结果: cat $RESULTS_DIR/evaluation_${STRATEGY}.json"
echo "  报告: cat $RESULTS_DIR/report_${STRATEGY}.txt"
echo ""
