#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON=${PYTHON:-python}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the final LAPS checkpoint}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the TRACE data root}
OUT=${OUT:?Set OUT to a fresh BO output directory}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
NGCM_TRAIN50=${NGCM_TRAIN50:-$ROOT/output/bo_calibration/ngcm_train50_seed2026/buffer.jsonl}
FINAL_STAGE=${CHECKPOINT%_fast}
VERTEX_TEST_RESULTS_DIR=${VERTEX_TEST_RESULTS_DIR:-$FINAL_STAGE/eval_before_fast_temp01_x8}
REUSE_VERTEX_TEST_RESULTS=${REUSE_VERTEX_TEST_RESULTS:-1}
BO_GPU_MEMORY_UTILIZATION=${BO_GPU_MEMORY_UTILIZATION:-0.88}
TASK_ORDER=${TASK_ORDER:-C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten}

if [[ ! -s "$NGCM_TRAIN50" || ! -s "$(dirname "$NGCM_TRAIN50")/buffer.manifest.json" ]]; then
  "$PYTHON" experiments/build_bo_calibration_buffer.py \
    --data-root "$DATA_ROOT" --task NumGLUE-cm --size 50 \
    --seed '2026:NumGLUE-cm:train50' \
    --output-dir "$(dirname "$NGCM_TRAIN50")"
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
IFS=',' read -r -a TASKS <<< "$TASK_ORDER"
if [[ ${#GPUS[@]} -ne ${#TASKS[@]} ]]; then
  echo "GPU_IDS must contain exactly eight devices" >&2
  exit 2
fi

mkdir -p "$OUT"
export PYTHONPATH="$ROOT:$ROOT/experiments${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

PIDS=()
vertex_test_args=()
if [[ "$REUSE_VERTEX_TEST_RESULTS" == "1" ]]; then
  vertex_test_args+=(--vertex-test-results-dir "$VERTEX_TEST_RESULTS_DIR")
fi
for I in "${!TASKS[@]}"; do
  TASK=${TASKS[$I]}
  GPU=${GPUS[$I]}
  CUDA_VISIBLE_DEVICES="$GPU" VLLM_USE_V1=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" -u experiments/search_trace_simplex_bo_ei_vllm.py \
      --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" --tasks "$TASK" \
      --budget 100 --task-budget MeetingBank=50 \
      --initial-points 16 --bo-batch 4 --candidate-pool 4096 \
      --ei-xi 0.05 --ei-stop 0.005 --early-stop-patience 3 \
      --probe-size 500 --max-num-seqs 192 \
      --validation-replay-buffer "$NGCM_TRAIN50" \
      --validation-replay-tasks NumGLUE-cm \
      --validation-temperature 0.1 --validation-samples 3 \
      --confirmation-fraction 0 --confirmation-top-k 3 \
      --confirmation-folds 1 --confirmation-full-validation \
      --include-task-vertex-finalist --confirmation-samples 8 \
      "${vertex_test_args[@]}" \
      --bootstrap-replicates 2000 --bootstrap-confidence 0.90 \
      --no-slow-fallback --heteroscedastic-noise --adaptive-xi \
      --gpu-memory-utilization "$BO_GPU_MEMORY_UTILIZATION" --seed 2026 \
      --output "$OUT/search_${TASK}.json" \
      >"$OUT/search_${TASK}.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || FAILED=1
done
(( FAILED == 0 )) || exit 1

"$PYTHON" - "$OUT" <<'PY'
import glob
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
merged = {
    "protocol": (
        "full selected validation probe BO (up to 500 official validation rows); "
        "NumGLUE-cm additionally includes a fixed "
        "50-row random training sample; Top-3 plus task vertex; fresh 8-run mean reranking"
    ),
    "tasks": {},
}
for path in glob.glob(str(root / "search_*.json")):
    merged["tasks"].update(json.load(open(path))["tasks"])
(root / "summary.json").write_text(json.dumps(merged, indent=2) + "\n")
PY

touch "$OUT/SEARCH_COMPLETE"
echo "Train-50 augmented Bayesian preference search complete: $OUT/summary.json"
