#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
export CUDA_HOME="${VLM_CUDA_HOME:-/usr/local/cuda}"
export PATH="$(dirname "${PYTHON}"):${CUDA_HOME}/bin:${PATH}"
export FLASHINFER_NVCC="${CUDA_HOME}/bin/nvcc"
export VLLM_USE_FLASHINFER_SAMPLER=1
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local Qwen3-VL checkpoint}"
export CUDA_VISIBLE_DEVICES="${VLM_GPU:-2}"

exec "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}" \
  --host 127.0.0.1 --port "${VLM_PORT:-8082}" \
  --gpu-memory-utilization 0.90 \
  --structured-outputs-config '{"backend":"xgrammar"}' \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --max-num-seqs 32 "$@"
