#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Rigid pipeline: nuScenes or generic frame indexes -> FLUX -> H/B0 -> asset library.
Usage: bash scripts/run_nuscenes.sh ACTION [--reuse-handoff] [frontend options]
Choose exactly one ACTION:
  --handoff-only          Extract and validate the complete handoff, then stop.
  --all-assets            Generate every object in the complete handoff.
  --asset-ids ID1,ID2      Generate exactly these obj_id values.
  --task-manifest FILE    Generate exactly these unchanged handoff JSON rows.
  --reuse-handoff         Use an existing run's handoff without extraction.
  --help                  Show help without starting models.
Frontend options such as --scene-ids scene-0103 pass to pipeline.py.
Use environment variables for shared settings:
  PYTHON / CONFIG         Orchestrator Python / frontend YAML (nuscenes or frames).
  GPUS / RUN_NAME         Frontend GPU CSV (4,5,6,7) / run name (rigid_nuscenes).
  OUTPUT_ROOT            Frontend root (project outputs); RUN_DIR=ROOT/runs/NAME.
  WORKERS_PER_GPU         Scene slots per GPU (1).
  GENERATION_CONFIG      Runtime/model JSON (configs/generation.example.json).
  GENERATION_GPUS         Generation GPU CSV (same as GPUS).
  RENDER_GPUS             Renderer GPU CSV (7).
  RENDER_PYTHON           Renderer environment (/home/wmj/miniconda3/envs/sim/bin/python).
  WORK_DIR               Jobs and evidence directory (RUN_DIR/work).
  LIBRARY_DIR            NEW final asset directory (RUN_DIR/library).
No default generation selection; full generation must be explicitly selected.
HELP
}
ACTION=""; SELECTION=""; REUSE=0; FRONTEND_ARGS=()
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --handoff-only|--all-assets|--asset-ids|--task-manifest)
      [[ -z "${ACTION}" ]] || { echo "Choose exactly one ACTION" >&2; exit 2; }
      ACTION="$1"; shift
      if [[ "${ACTION}" == --asset-ids || "${ACTION}" == --task-manifest ]]; then
        (($#)) && [[ -n "$1" && "$1" != --* ]] || { echo "${ACTION} needs a value" >&2; exit 2; }
        SELECTION="$1"; shift
      fi ;;
    --reuse-handoff) REUSE=1; shift ;;
    --config|--gpu|--run-name|--output-root|--scene-workers-per-gpu|--dry-run)
      echo "Use CONFIG/GPUS/RUN_NAME/OUTPUT_ROOT/WORKERS_PER_GPU variables; run pipeline.py directly for --dry-run." >&2; exit 2 ;;
    *) FRONTEND_ARGS+=("$1"); shift ;;
  esac
done
[[ -n "${ACTION}" ]] || { usage >&2; exit 2; }
[[ "${REUSE}" == 0 || ${#FRONTEND_ARGS[@]} == 0 ]] || { echo "Frontend options do not apply with --reuse-handoff" >&2; exit 2; }
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON="${PYTHON:-/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python}"
CONFIG="${CONFIG:-configs/nuscenes.yaml}"
GPUS="${GPUS:-4,5,6,7}"
RUN_NAME="${RUN_NAME:-rigid_nuscenes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_NAME}"
WORK_DIR="${WORK_DIR:-${RUN_DIR}/work}"
LIBRARY_DIR="${LIBRARY_DIR:-${RUN_DIR}/library}"
GENERATION_CONFIG="${GENERATION_CONFIG:-configs/generation.example.json}"
GENERATION_GPUS="${GENERATION_GPUS:-${GPUS}}"
RENDER_GPUS="${RENDER_GPUS:-7}"
RENDER_PYTHON="${RENDER_PYTHON:-/home/wmj/miniconda3/envs/sim/bin/python}"
if [[ "${ACTION}" == --task-manifest && ! -f "${SELECTION}" ]]; then
  echo "Task manifest does not exist: ${SELECTION}" >&2; exit 2
fi
if [[ "${ACTION}" != --handoff-only ]]; then
  [[ ! -e "${LIBRARY_DIR}" ]] || { echo "Final library exists; choose a new LIBRARY_DIR" >&2; exit 2; }
  [[ "${GENERATION_GPUS}" =~ ^[0-9]+(,[0-9]+)*$ && "${RENDER_GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "GENERATION_GPUS and RENDER_GPUS must be comma-separated GPU indices" >&2; exit 2;
  }
  [[ -f "${GENERATION_CONFIG}" && -x "${RENDER_PYTHON}" ]] || { echo "Missing generation config or renderer Python" >&2; exit 2; }
fi
if [[ "${REUSE}" == 0 ]]; then
  "${PYTHON}" -m asset_extraction.vlm --check-ready --config "${CONFIG}"
  "${PYTHON}" -m asset_extraction.pipeline --config "${CONFIG}" --gpu "${GPUS}" --run-name "${RUN_NAME}" \
    --output-root "${OUTPUT_ROOT}" --scene-workers-per-gpu "${WORKERS_PER_GPU}" "${FRONTEND_ARGS[@]}"
  "${PYTHON}" -m nuscenes.quality_filter "${RUN_DIR}/merged/manifest.json" --run-dir "${RUN_DIR}" \
    --output "${RUN_DIR}/merged/manifest.filtered.json" --reject-log "${RUN_DIR}/logs/rigid_quality_rejects.jsonl"
  "${PYTHON}" -m nuscenes.prepare_handoff "${RUN_DIR}/merged/manifest.filtered.json" --output "${RUN_DIR}/handoff/flux_input.json"
  "${PYTHON}" -m nuscenes.prepare_routes --handoff "${RUN_DIR}/handoff/flux_input.json" \
    --run-dir "${RUN_DIR}" --output "${RUN_DIR}/handoff/asset_routes.json"
fi
"${PYTHON}" -m nuscenes.validate_handoff "${RUN_DIR}/handoff/flux_input.json"
echo "Complete handoff: ${RUN_DIR}/handoff/flux_input.json"
[[ "${ACTION}" != --handoff-only ]] || exit 0
"${PYTHON}" - "${RUN_DIR}/handoff/flux_input.json" "${WORK_DIR}/selected" "${ACTION}" "${SELECTION}" <<'PY'
import json, sys
from pathlib import Path
from asset_extraction.io_utils import write_json
source, directory, action, selection = sys.argv[1:]
rows = json.loads(Path(source).read_text())
index = {row['obj_id']: row for row in rows}
if len(index) != len(rows):
    raise ValueError('Duplicate identity in complete handoff')
if action == '--task-manifest':
    chosen = json.loads(Path(selection).read_text())
    if not isinstance(chosen, list) or any(not isinstance(row, dict) or row != index.get(row.get('obj_id')) for row in chosen):
        raise ValueError('Task manifest must contain unchanged rows from the complete handoff')
elif action == '--asset-ids':
    ids = selection.split(',')
    if len(ids) != len(set(ids)) or any(uid not in index for uid in ids):
        raise ValueError('Asset IDs must be distinct and present in the complete handoff')
    chosen = [index[uid] for uid in ids]
else:
    chosen = rows
if not chosen or len({row['obj_id'] for row in chosen}) != len(chosen):
    raise ValueError('Selected query set is empty or contains duplicate identities')
write_json(Path(directory) / 'flux_input.json', chosen)
audit = dict(complete_queries=len(rows), selected_queries=len(chosen), action=action,
             selected_obj_ids=[row['obj_id'] for row in chosen])
write_json(Path(directory) / 'selection.json', audit)
print(json.dumps(audit, ensure_ascii=False))
PY
"${PYTHON}" -m nuscenes.prepare_routes --handoff "${WORK_DIR}/selected/flux_input.json" \
  --run-dir "${RUN_DIR}" --output "${WORK_DIR}/selected/asset_routes.json"
IFS=',' read -r -a GENERATION_GPU_IDS <<< "${GENERATION_GPUS}"
IFS=',' read -r -a RENDER_GPU_IDS <<< "${RENDER_GPUS}"
"${PYTHON}" -m asset_extraction.generation \
  --handoff "${WORK_DIR}/selected/flux_input.json" --routes "${WORK_DIR}/selected/asset_routes.json" \
  --config "${GENERATION_CONFIG}" --work "${WORK_DIR}/generation" --gpus "${GENERATION_GPU_IDS[@]}"
"${PYTHON}" -m asset_extraction.export --generation "${WORK_DIR}/generation/generation.json" --config "${CONFIG}" \
  --output "${LIBRARY_DIR}" --work "${WORK_DIR}/curation" --renderer-python "${RENDER_PYTHON}" --gpus "${RENDER_GPU_IDS[@]}"
echo "Asset library: ${LIBRARY_DIR}/manifests/asset_library_rigid.json"
