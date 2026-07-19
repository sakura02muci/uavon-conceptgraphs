#!/usr/bin/env bash
# Repeat a compact, varied Barnyard ObjectNav set under a render-quality gate.
# Run from UAV_ON; all outputs are independent per repeat and therefore safe
# to resume/review without mixing trajectories from different trials.
set -euo pipefail

REPEATS="${1:-3}"
GPU_INDEX="${UAVON_GPU_INDEX:-4}"
EPISODES="${UAVON_REPEAT_EPISODES:-60,23,36,92,75}"
RUN_TAG="${UAVON_REPEAT_TAG:-v19_repeat}"

for repeat in $(seq 1 "$REPEATS"); do
  output="results/uavon/${RUN_TAG}_${repeat}_gpu${GPU_INDEX}.json"
  log="results/uavon/${RUN_TAG}_${repeat}_gpu${GPU_INDEX}.log"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH=src python -u scripts/eval_simple_uavon.py \
    --dataset ../UAV-ON-dataset/valset/Barnyard.json \
    --episode-ids "$EPISODES" \
    --strategy hierarchical --detector groundingdino --max-steps 60 \
    --output "$output" \
    --graph-dir "results/uavon/scene_graphs_${RUN_TAG}_${repeat}_gpu${GPU_INDEX}" \
    --diagnostic-dir "results/uavon/diagnostics_${RUN_TAG}_${repeat}_gpu${GPU_INDEX}" \
    --clip-crop-verify --map-clip-features --target-memory-mode conservative \
    --safe-step-mode --collision-recovery --visual-close-approach \
    --render-warmup-frames 100 --render-warmup-delay 0.2 \
    --render-quality-max-black-ratio 0.03 \
    --render-quality-max-spread 0.01 \
    --render-quality-retry-rounds 2 \
    --resume > "$log" 2>&1
done
