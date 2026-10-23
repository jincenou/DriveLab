#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python}"
CONFIG="${CONFIG:-configs/nuscenes.yaml}"
GPUS="${GPUS:-4,5,6,7}"
RUN_NAME="${RUN_NAME:-rigid_nuscenes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"

"${PYTHON}" -m asset_extraction.vlm --check-ready --config "${CONFIG}"
"${PYTHON}" -m asset_extraction.pipeline \
  --config "${CONFIG}" --gpu "${GPUS}" --run-name "${RUN_NAME}" \
  --output-root "${OUTPUT_ROOT}" --scene-workers-per-gpu "${WORKERS_PER_GPU}" "$@"

RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_NAME}"
"${PYTHON}" -m nuscenes.quality_filter "${RUN_DIR}/merged/manifest.json" \
  --output "${RUN_DIR}/merged/manifest.filtered.json" \
  --reject-log "${RUN_DIR}/logs/rigid_quality_rejects.jsonl"
"${PYTHON}" -m nuscenes.prepare_handoff "${RUN_DIR}/merged/manifest.filtered.json" \
  --output "${RUN_DIR}/handoff/flux_input.json"
"${PYTHON}" -m nuscenes.prepare_routes \
  --handoff "${RUN_DIR}/handoff/flux_input.json" --run-dir "${RUN_DIR}" \
  --output "${RUN_DIR}/handoff/asset_routes.json"

echo "FLUX handoff: ${RUN_DIR}/handoff/flux_input.json"
