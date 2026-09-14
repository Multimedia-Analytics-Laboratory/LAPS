#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
TORCHRUN=${TORCHRUN:-torchrun}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
BO_GPU_IDS=${BO_GPU_IDS:-0,1,2,3,4,5,6,7}
REPLAY_GLOBAL_BATCH=${REPLAY_GLOBAL_BATCH:-8}
MODEL=${MODEL:-}
DATA_ROOT=${DATA_ROOT:-}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/output/laps}
GOLD_ROOT=${GOLD_ROOT:-$ROOT/replay_buffers/gold_rho_0.01}
SLOW_REPLAY_SFT=${SLOW_REPLAY_SFT:-0}
REBASE_LR=${REBASE_LR:-0.01}
PROMPT_LENGTH=${PROMPT_LENGTH:-30}
BEZIER_DEGREE=${BEZIER_DEGREE:-3}
TASK_ORDER=${TASK_ORDER:-C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten}
PROMOT_PER_DEVICE_BATCH=${PROMOT_PER_DEVICE_BATCH:-4}
EVAL_INTERMEDIATE_STAGES=${EVAL_INTERMEDIATE_STAGES:-1}
EVAL_SLOW_BASELINE=${EVAL_SLOW_BASELINE:-1}
EVAL_DIAGONAL_ONLY=${EVAL_DIAGONAL_ONLY:-0}
BO_REUSE_VERTEX_TEST_RESULTS=${BO_REUSE_VERTEX_TEST_RESULTS:-1}

usage() {
  cat <<'EOF'
Usage: scripts/run_laps.sh --model PATH --data-root PATH [options]

Required:
  --model PATH          Base Hugging Face checkpoint
  --data-root PATH      TRACE LLM-CL-Benchmark_5000 directory

Options:
  --output-dir PATH     Run directory (default: output/laps)
  --gold-root PATH      Stage-wise gold replay buffers
  --slow-replay-sft     Also update plain Slow on gold replay (ablation)
  --rebase-lr FLOAT     Historical vertex rebase learning rate (default: 0.01)
  --prompt-length INT   Number of learned soft-prompt tokens (default: 30)
  --bezier-degree INT   Bezier simplex degree k (default: 3)
  --task-order LIST     Comma-separated TRACE task order (default: canonical)
  --promot-per-device-batch INT
                        Maximum ProMoT batch per GPU; OOM falls back through
                        16,8,4,2,1 while preserving global batch 128 (default: 4)
  --final-eval-only     Skip stage 0--6 lower-triangular evaluations
  --diagonal-only-eval  Evaluate only the newly learned task vertex at each stage
  --no-slow-eval        Evaluate only the task vertices, not Slow baselines
  --gpu-ids LIST        Eight training GPUs (default: 0,1,2,3,4,5,6,7)
  --bo-gpu-ids LIST     Eight GPUs used by final BO (default: 0,...,7)
  --python PATH         Python executable
  --torchrun PATH       torchrun executable

The launcher is resumable. Completed stages and evaluations are skipped.
EOF
}

while (($#)); do
  case "$1" in
    --model) MODEL=$2; shift 2 ;;
    --data-root) DATA_ROOT=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --gold-root) GOLD_ROOT=$2; shift 2 ;;
    --slow-replay-sft) SLOW_REPLAY_SFT=1; shift ;;
    --rebase-lr) REBASE_LR=$2; shift 2 ;;
    --prompt-length) PROMPT_LENGTH=$2; shift 2 ;;
    --bezier-degree) BEZIER_DEGREE=$2; shift 2 ;;
    --task-order) TASK_ORDER=$2; shift 2 ;;
    --promot-per-device-batch) PROMOT_PER_DEVICE_BATCH=$2; shift 2 ;;
    --final-eval-only) EVAL_INTERMEDIATE_STAGES=0; shift ;;
    --diagonal-only-eval)
      EVAL_DIAGONAL_ONLY=1
      EVAL_INTERMEDIATE_STAGES=1
      EVAL_SLOW_BASELINE=0
      BO_REUSE_VERTEX_TEST_RESULTS=0
      shift
      ;;
    --no-slow-eval) EVAL_SLOW_BASELINE=0; shift ;;
    --gpu-ids) GPU_IDS=$2; shift 2 ;;
    --bo-gpu-ids) BO_GPU_IDS=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --torchrun) TORCHRUN=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "--model is required" >&2; exit 2; }
[[ -n "$DATA_ROOT" ]] || { echo "--data-root is required" >&2; exit 2; }
[[ -d "$DATA_ROOT" ]] || { echo "Data root not found: $DATA_ROOT" >&2; exit 2; }
[[ -d "$GOLD_ROOT" ]] || { echo "Replay root not found: $GOLD_ROOT" >&2; exit 2; }
[[ "$BEZIER_DEGREE" =~ ^[1-9][0-9]*$ ]] || { echo "--bezier-degree must be a positive integer" >&2; exit 2; }
case "$PROMOT_PER_DEVICE_BATCH" in
  1|2|4|8|16) ;;
  *) echo "--promot-per-device-batch must be one of 1,2,4,8,16" >&2; exit 2 ;;
esac

# Evaluation workers intentionally chdir to /tmp before importing vLLM.  Keep
# every cross-process filesystem argument absolute so relative CLI paths do
# not silently become invalid after that boundary.
DATA_ROOT=$(realpath "$DATA_ROOT")
GOLD_ROOT=$(realpath "$GOLD_ROOT")
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")
if [[ -d "$MODEL" ]]; then MODEL=$(realpath "$MODEL"); fi

mkdir -p "$OUTPUT_DIR/logs"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export TRACE_TASK_ORDER=$TASK_ORDER
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

env ROOT="$ROOT" PYTHON="$PYTHON" TORCHRUN="$TORCHRUN" \
  DATA_ROOT="$DATA_ROOT" GOLD_ROOT="$GOLD_ROOT" RUN="$OUTPUT_DIR" \
  START_STAGE=0 INITIAL_MODEL="$MODEL" \
  REPLAY_GLOBAL_BATCH="$REPLAY_GLOBAL_BATCH" GPU_IDS="$GPU_IDS" \
  SLOW_REPLAY_SFT="$SLOW_REPLAY_SFT" \
  EVAL_INTERMEDIATE_STAGES="$EVAL_INTERMEDIATE_STAGES" \
  EVAL_SLOW_BASELINE="$EVAL_SLOW_BASELINE" \
  EVAL_DIAGONAL_ONLY="$EVAL_DIAGONAL_ONLY" \
  BO_REUSE_VERTEX_TEST_RESULTS="$BO_REUSE_VERTEX_TEST_RESULTS" \
  BO_GPU_IDS="$BO_GPU_IDS" \
  BO_SCRIPT="$ROOT/scripts/run_simplex_bo_top3_vertex_repeat8_ngcm_train50.sh" \
  BO_OUT_NAME="simplex_bo_top3_vertex_repeat8_ngcm_train50" \
  PROMPT_LENGTH="$PROMPT_LENGTH" BEZIER_DEGREE="$BEZIER_DEGREE" \
  TASK_ORDER="$TASK_ORDER" TRACE_TASK_ORDER="$TASK_ORDER" \
  PROMPT_LR=0.3 SLOW_LR=1e-5 REBASE_LR="$REBASE_LR" \
  POST_SLOW_VERTEX_LR=0.03 PROMOT_PER_DEVICE_BATCH="$PROMOT_PER_DEVICE_BATCH" \
  bash "$ROOT/scripts/run_laps_full_stream.sh"
