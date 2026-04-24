#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORKSPACE_ROOT=$( cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd )
export CODE_ROOT="${CODE_ROOT:-${WORKSPACE_ROOT}}"
DEFAULT_CODE_SRC=${OLMO_CORE_SRC:-"${WORKSPACE_ROOT}/OLMo-core/src"}
UBDATALOADER_SRC="${WORKSPACE_ROOT}/packages/ubdataloader/src"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OLMO_SHARED_FS="${OLMO_SHARED_FS:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"
export LANG="${LANG:-en_US.UTF-8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export NCCL_ALGO="${NCCL_ALGO:-^NVLS}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"

: "${WANDB_API_KEY:=}"
: "${WANDB_BASE_URL:=}"
: "${WANDB_PROJECT:=}"
: "${WANDB_ENTITY:=}"
: "${WANDB_MODE:=online}"
export WANDB_API_KEY WANDB_BASE_URL WANDB_PROJECT WANDB_ENTITY WANDB_MODE

if [[ -d "${DEFAULT_CODE_SRC}" ]]; then
  LOCAL_PYTHONPATH="${DEFAULT_CODE_SRC}"
  if [[ -d "${UBDATALOADER_SRC}" ]]; then
    LOCAL_PYTHONPATH="${LOCAL_PYTHONPATH}:${UBDATALOADER_SRC}"
  fi
  export PYTHONPATH="${LOCAL_PYTHONPATH}:${PYTHONPATH:-}"
fi

cd "${WORKSPACE_ROOT}"

if [[ -n "${INSTALL_FROM_EXISTING_SH:-}" ]]; then
  echo "[info] before install hook: $(date) pwd=$(pwd)"
  bash "${INSTALL_FROM_EXISTING_SH}"
  echo "[info] after install hook:  $(date) rc=$? pwd=$(pwd)"
fi

YAML_CONFIG=${1:-${YAML_CONFIG:-"${WORKSPACE_ROOT}/configs/xgram_3h_2v_hash.yaml"}}
EXPERIMENT_NAME="$(basename "${YAML_CONFIG}" .yaml)"

# Recover the default run name from YAML so shell-side checkpoint staging matches the old launcher.
if [[ -z "${RUN_NAME:-}" ]] && [[ -f "${YAML_CONFIG}" ]]; then
  RUN_NAME="$(
    python3 - "${YAML_CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
run_cfg = cfg.get("run") or {}
value = run_cfg.get("run_name") or run_cfg.get("experiment_name") or ""
sys.stdout.write(str(value))
PY
  )"
fi

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${PET_NNODES:-${SLURM_NNODES:-${NNODES:-1}}}
export NODE_RANK=${PET_NODE_RANK:-${SLURM_PROCID:-${NODE_RANK:-0}}}
if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ "${NNODES}" -gt 1 ]]; then
    echo "Error: multi-node launch detected (NNODES=${NNODES}) but MASTER_ADDR is missing."
    exit 1
  else
    export MASTER_ADDR=localhost
  fi
else
  export MASTER_ADDR
fi
export MASTER_PORT=${MASTER_PORT:-6000}
export LAUNCH_TIME_TAG=${LAUNCH_TIME_TAG:-$(date +%Y%m%d-%H%M%S)}

PROJ_ROOT=${PROJ_ROOT:-}
if [[ -z "${CKPT_SRC:-}" ]]; then
  if [[ -n "${PROJ_ROOT}" ]] && [[ -n "${RUN_NAME:-}" ]]; then
    CKPT_SRC="${PROJ_ROOT}/${RUN_NAME}/dataloader.ckpt"
  elif [[ -n "${PROJ_ROOT}" ]]; then
    CKPT_SRC="${PROJ_ROOT}/dataloader.ckpt"
  else
    CKPT_SRC=""
  fi
fi

if [[ -n "${CKPT_SRC}" ]] && [[ -f "${CKPT_SRC}.lock" ]]; then
  echo "[info] Removing stale DuckDB lock file: ${CKPT_SRC}.lock"
  rm -f "${CKPT_SRC}.lock" || true
fi

TMP_ROOT=${TMPDIR:-/tmp}
TMP_PARENT="$TMP_ROOT/olmo_stream_ckpts"
mkdir -p "$TMP_PARENT"

RUN_ID="${LAUNCH_TIME_TAG}-$RANDOM-$$"
RUN_DIR="$TMP_PARENT/run-${RUN_ID}"
mkdir -p "$RUN_DIR"
RUN_CKPT="$RUN_DIR/dataloader.ckpt"

if [[ -n "${CKPT_SRC}" ]] && [[ -f "$CKPT_SRC" ]]; then
  echo "[info] Copying ckpt from: $CKPT_SRC -> $RUN_CKPT"
  cp -p "$CKPT_SRC" "$RUN_CKPT" 2>/dev/null || cp "$CKPT_SRC" "$RUN_CKPT"
elif [[ -n "${CKPT_SRC}" ]]; then
  echo "[warn] Source ckpt not found: $CKPT_SRC ; DuckDB will create a new database on first use"
else
  echo "[warn] CKPT_SRC is empty; set PROJ_ROOT or CKPT_SRC to reuse an existing DuckDB checkpoint"
fi

DISTRIBUTED_ARGS=(
  --nproc_per_node ${GPUS_PER_NODE}
  --nnodes ${NNODES}
  --node_rank ${NODE_RANK}
  --master_addr ${MASTER_ADDR}
  --master_port ${MASTER_PORT}
)

LOG_DIR=${LOG_DIR:-./logs}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d-%H%M%S)_${EXPERIMENT_NAME}.log"
ACTUAL_LOG_FILE="${LOG_FILE%.*}_rank${NODE_RANK:-$(hostname)}.log"

EXTRA_ARGS=()
if [ -n "${RUN_NAME:-}" ]; then
  EXTRA_ARGS+=(--run-name "${RUN_NAME}")
fi
if [ -n "${SAVE_ROOT:-}" ]; then
  EXTRA_ARGS+=(--save-root "${SAVE_ROOT}")
fi
if [ -n "${MICRO_BATCH_SIZE:-}" ]; then
  EXTRA_ARGS+=(--micro-batch-size "${MICRO_BATCH_SIZE}")
fi
if [ -n "${STREAMING_DATA_PATH:-}" ]; then
  read -r -a STREAMING_DATA_PATH_ARGS <<< "${STREAMING_DATA_PATH}"
  EXTRA_ARGS+=(--streaming-data-path "${STREAMING_DATA_PATH_ARGS[@]}")
fi
if [ -n "${STREAMING_TOKENIZER_MODEL:-}" ]; then
  EXTRA_ARGS+=(--streaming-tokenizer-model "${STREAMING_TOKENIZER_MODEL}")
fi
if [ -n "${STREAMING_USE_TOKEN_COLUMN:-}" ]; then
  EXTRA_ARGS+=(--streaming-use-token-column "${STREAMING_USE_TOKEN_COLUMN}")
fi
if [ -n "${TEXT_CHUNK_SIZE:-}" ]; then
  EXTRA_ARGS+=(--streaming-text-chunk-size "${TEXT_CHUNK_SIZE}")
fi
if [ -n "${PREFETCH_QUEUE_SIZE:-}" ]; then
  EXTRA_ARGS+=(--streaming-prefetch-queue-size "${PREFETCH_QUEUE_SIZE}")
fi
if [ -n "${STREAMING_PACK_METHOD:-}" ]; then
  EXTRA_ARGS+=(--streaming-pack-method "${STREAMING_PACK_METHOD}")
fi

set -x

echo "[info] Run name: ${RUN_NAME:-<from-yaml>}"
echo "[info] Master port: ${MASTER_PORT}"
echo "[info] YAML config: ${YAML_CONFIG}"
if [ -n "${STREAMING_DATA_PATH:-}" ]; then
  echo "[info] Streaming data path source: env STREAMING_DATA_PATH"
else
  echo "[info] Streaming data path source: python launcher defaults / yaml"
fi

torchrun "${DISTRIBUTED_ARGS[@]}" \
  "${SCRIPT_DIR}/olmo_train.py" \
  --config "${YAML_CONFIG}" \
  --launcher-mode shell_compatible \
  --streaming-ckpt-path "${RUN_CKPT}" \
  "${EXTRA_ARGS[@]}" \
  </dev/null \
  > "${ACTUAL_LOG_FILE}" 2>&1

echo "[TEST] TRAIN_LAUNCHED log=${ACTUAL_LOG_FILE}"
