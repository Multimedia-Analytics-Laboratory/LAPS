#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to TRACE LLM-CL-Benchmark_5000}
OUT=${OUT:-$ROOT/replay_buffers/gold_rho_0.01}
TASK_ORDER=${TASK_ORDER:-C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten}

IFS=',' read -r -a TASKS <<< "$TASK_ORDER"
if [[ ${#TASKS[@]} -ne 8 ]]; then
  echo "TASK_ORDER must contain exactly eight comma-separated tasks" >&2
  exit 2
fi

for stage in 0 1 2 3 4 5 6; do
  task=${TASKS[$stage]}
  output="$OUT/$(printf 'stage_%02d_%s' "$stage" "$task")/buffer.jsonl"
  TRACE_TASK_ORDER="$TASK_ORDER" "$PYTHON" \
    "$ROOT/experiments/build_gold_replay_buffer.py" \
    --data-root "$DATA_ROOT" --stage "$stage" --buffer-size 50 \
    --seed 42 --include-task --output "$output"
done

echo "Replay buffers written to $OUT"
