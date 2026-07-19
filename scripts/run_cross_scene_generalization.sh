#!/usr/bin/env bash
# Fixed-seed, cross-scene generalization evaluation for UAV-ON validation data.
# It samples one episode per validation scene (14 total) and uses a fresh UE4
# process per scene, avoiding accidental reuse of the previous scene's map.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT="$ROOT/UAV_ON"
DATA_ROOT="$ROOT/UAV-ON-dataset/valset"
GPU_INDEX="${UAVON_GPU_INDEX:-4}"
UE4_ADAPTER_INDEX=$((GPU_INDEX + 1))
GPU_MAX_UTILIZATION="${UAVON_GPU_MAX_UTILIZATION:-70}"
GPU_MIN_FREE_MB="${UAVON_GPU_MIN_FREE_MB:-32000}"
# Set this to a positive value only when it is acceptable for a scheduled run
# to wait for a shared GPU. The default fails fast rather than starting UE4 on
# a saturated device and producing invalid render-gate rejections.
GPU_WAIT_SECONDS="${UAVON_GPU_WAIT_SECONDS:-0}"
GPU_RETRY_SECONDS="${UAVON_GPU_RETRY_SECONDS:-30}"
SEED="${UAVON_GENERALIZATION_SEED:-20260718}"
RUN_TAG="${UAVON_GENERALIZATION_TAG:-generalization_v20}"
MAX_STEPS="${UAVON_GENERALIZATION_MAX_STEPS:-80}"
SAMPLES_PER_SCENE="${UAVON_GENERALIZATION_SAMPLES_PER_SCENE:-1}"
SCENE_FILTER="${UAVON_GENERALIZATION_SCENES:-}"
RESULT_DIR="$PROJECT/results/uavon/$RUN_TAG"
LOG_DIR="$RESULT_DIR/logs"

mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/uavon-matplotlib}"

gpu_is_ready() {
  local values free_mb utilization
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU preflight failed: nvidia-smi is unavailable." >&2
    return 2
  fi
  values="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" || {
    echo "GPU preflight failed: cannot query GPU $GPU_INDEX." >&2
    return 2
  }
  IFS=, read -r free_mb utilization <<< "$values"
  free_mb="${free_mb//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  if [[ ! "$free_mb" =~ ^[0-9]+$ || ! "$utilization" =~ ^[0-9]+$ ]]; then
    echo "GPU preflight failed: unexpected nvidia-smi output: $values" >&2
    return 2
  fi
  echo "GPU $GPU_INDEX preflight: free=${free_mb}MiB, utilization=${utilization}% (requires >=${GPU_MIN_FREE_MB}MiB and <=${GPU_MAX_UTILIZATION}%)."
  (( free_mb >= GPU_MIN_FREE_MB && utilization <= GPU_MAX_UTILIZATION ))
}

waited_seconds=0
until gpu_is_ready; do
  if (( waited_seconds >= GPU_WAIT_SECONDS )); then
    echo "Refusing to start evaluation on busy GPU $GPU_INDEX. Set UAVON_GPU_WAIT_SECONDS to wait, or explicitly override the thresholds if appropriate." >&2
    exit 2
  fi
  sleep "$GPU_RETRY_SECONDS"
  waited_seconds=$((waited_seconds + GPU_RETRY_SECONDS))
done

SCENES=(
  "Barnyard:TEST_ENVS/Barnyard/Barnyard_test1.sh"
  "BrushifyRoad:TEST_ENVS/BrushifyRoad/BrushifyRoad_test1.sh"
  "BrushifyUrban:TEST_ENVS/BrushifyUrban/BrushifyUrban.sh"
  "CabinLake:TEST_ENVS/CabinLake/CabinLake.sh"
  "CityPark:TEST_ENVS/CityPark/CityPark.sh"
  "CityStreet:TEST_ENVS/CityStreet/CleanCityStreet.sh"
  "DownTown:TEST_ENVS/DownTown/DownTown_test1.sh"
  "NYC:TEST_ENVS/NYC/NYC1950.sh"
  "Neighborhood:TEST_ENVS/Neighborhood/NewNeighborhood.sh"
  "Slum:TEST_ENVS/Slum/Slum_test1.sh"
  "UrbanJapan:TEST_ENVS/UrbanJapan/UrbanJapan.sh"
  "Venice:TEST_ENVS/Venice/Vinice_test1.sh"
  "WesternTown:TEST_ENVS/WesternTown/WesternTown_test1.sh"
  "WinterTown:TEST_ENVS/WinterTown/WinterTown_test1.sh"
)

SCENE_PID=""
cleanup_scene() {
  if [[ -n "${SCENE_PID}" ]] && kill -0 "$SCENE_PID" 2>/dev/null; then
    kill -- "-$SCENE_PID" 2>/dev/null || true
    wait "$SCENE_PID" 2>/dev/null || true
  fi
  SCENE_PID=""
}
trap cleanup_scene EXIT INT TERM

wait_for_airsim() {
  local attempts=0
  until python - <<'PY'
import airsim
try:
    client = airsim.MultirotorClient(ip='127.0.0.1', port=41451)
    client.confirmConnection()
except Exception:
    raise SystemExit(1)
PY
  do
    attempts=$((attempts + 1))
    if (( attempts >= 90 )); then
      return 1
    fi
    sleep 2
  done
}

for index in "${!SCENES[@]}"; do
  IFS=: read -r scene launcher_rel <<< "${SCENES[$index]}"
  if [[ -n "$SCENE_FILTER" && ",${SCENE_FILTER}," != *",${scene},"* ]]; then
    continue
  fi
  dataset="$DATA_ROOT/$scene.json"
  launcher="$ROOT/$launcher_rel"
  output="$RESULT_DIR/$scene.json"
  scene_log="$LOG_DIR/$scene.ue4.log"
  eval_log="$LOG_DIR/$scene.eval.log"

  episode_ids="$(python - "$dataset" "$SEED" "$index" "$SAMPLES_PER_SCENE" <<'PY'
import json, random, sys
episodes = json.load(open(sys.argv[1]))
rng = random.Random(int(sys.argv[2]) + int(sys.argv[3]))
count = min(max(1, int(sys.argv[4])), len(episodes))
selected = rng.sample(episodes, count)
print(','.join(str(episode['episode_id']) for episode in selected))
PY
)"
  IFS=, read -ra selected_ids <<< "$episode_ids"
  for episode_id in "${selected_ids[@]}"; do
    printf '%s\t%s\t%s\n' "$scene" "$episode_id" "$dataset" >> "$RESULT_DIR/sample_manifest.tsv"
  done

  echo "[$(date -Is)] scene=$scene episode_ids=$episode_ids"
  setsid bash "$launcher" -RenderOffscreen -NoSound -NoSplash -NoVSync \
    "-GraphicsAdapter=$UE4_ADAPTER_INDEX" -ResX=1024 -ResY=576 \
    '-ExecCmds=r.Streaming.PoolSize 16384,r.Streaming.UseFixedPoolSize 1' \
    >"$scene_log" 2>&1 &
  SCENE_PID=$!

  if ! wait_for_airsim; then
    echo "AirSim did not become ready for $scene" | tee -a "$eval_log"
    cleanup_scene
    continue
  fi

  set +e
  (
    cd "$PROJECT"
    python -u scripts/eval_simple_uavon.py \
      --dataset "$dataset" --episode-ids "$episode_ids" \
      --strategy hierarchical --detector groundingdino --disable-llm \
      --clip-crop-verify --map-clip-features --max-steps "$MAX_STEPS" \
      --safe-step-mode --collision-recovery --visual-close-approach \
      --render-warmup-frames 100 --render-quality-max-black-ratio 0.03 \
      --render-quality-max-spread 0.01 --render-quality-retry-rounds 2 \
      --output "$output" --diagnostic-dir "$RESULT_DIR/diagnostics/$scene"
  ) 2>&1 | tee "$eval_log"
  eval_status=${PIPESTATUS[0]}
  set -e
  if (( eval_status != 0 )); then
    echo "[$(date -Is)] evaluator exited with status=$eval_status for scene=$scene" | tee -a "$eval_log"
  fi
  cleanup_scene
  sleep 5
done

python - "$RESULT_DIR" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(root, '*.json'))):
    with open(path) as f:
        results = json.load(f)
    for r in results:
        rows.append({
            'scene': os.path.splitext(os.path.basename(path))[0],
            'episode_id': r.get('episode_id'), 'target': r.get('target'),
            'success': r.get('success'), 'min_distance_to_goal': r.get('min_distance_to_goal'),
            'failure_type': r.get('failure_type'), 'success_reason': r.get('success_reason'),
            'steps': r.get('steps'), 'target_evidence_stats': r.get('target_evidence_stats'),
        })
summary = {
    'seed': os.environ.get('UAVON_GENERALIZATION_SEED', '20260718'),
    'num_scenes': len(rows),
    'successes': sum(bool(r['success']) for r in rows),
    'success_rate': sum(bool(r['success']) for r in rows) / len(rows) if rows else 0,
    'episodes': rows,
}
with open(os.path.join(root, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY
