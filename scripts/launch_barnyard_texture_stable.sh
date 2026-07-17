#!/usr/bin/env bash
# Launch the packaged Barnyard environment with a fixed, larger UE4 texture pool.
# Extra Unreal flags can be appended by the caller.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POOL_MB="${UAVON_TEXTURE_POOL_MB:-8192}"
RES_X="${UAVON_RES_X:-1024}"
RES_Y="${UAVON_RES_Y:-576}"
# Physical GPU numbering is one-based in our experiment notes: GPU 3 is CUDA
# device index 2.  Keep an override for shared-node scheduling.
GPU_INDEX="${UAVON_GPU_INDEX:-2}"
# UE4's Vulkan GraphicsAdapter argument is one-based, unlike CUDA_VISIBLE_DEVICES.
UE4_ADAPTER_INDEX=$((GPU_INDEX + 1))
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

echo "Launching Barnyard on physical GPU $UE4_ADAPTER_INDEX (CUDA index $GPU_INDEX; UE4 adapter $UE4_ADAPTER_INDEX)"

exec "$ROOT/TEST_ENVS/Barnyard/Barnyard_test1.sh" \
  -RenderOffscreen -NoSound -NoSplash "-GraphicsAdapter=${UE4_ADAPTER_INDEX}" \
  "-ResX=${RES_X}" "-ResY=${RES_Y}" \
  "-ExecCmds=r.Streaming.PoolSize ${POOL_MB},r.Streaming.UseFixedPoolSize 1" \
  "$@"
