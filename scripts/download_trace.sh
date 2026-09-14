#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
USE_PROXY=${USE_PROXY:-0}
ARCHIVE=${ARCHIVE:-$ROOT/data/TRACE-Benchmark.zip}
FILE_ID=1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV
mkdir -p "$ROOT/data"
if [[ $USE_PROXY == 0 ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi
if [[ ! -s "$ARCHIVE" ]]; then
  "$PYTHON" -m gdown "$FILE_ID" -O "$ARCHIVE"
fi
unzip -q -o "$ARCHIVE" -d "$ROOT/data"
"$PYTHON" "$ROOT/experiments/prepare_trace_opr_msswift_full.py" \
  --source-root "$ROOT/data/TRACE-Benchmark/LLM-CL-Benchmark_5000" \
  --output-root "$ROOT/data/trace_jsonl"
echo "TRACE is ready under $ROOT/data"
