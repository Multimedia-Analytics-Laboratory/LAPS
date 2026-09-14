## Method

At continual stage `t`, LAPS performs four steps:

1. Learn the current task vertex, then update the slow model with completion-only SFT.
2. Recalibrate the current vertex and transport historical vertices with token-level forward KL from the preceding-stage model.
3. Freeze the slow model and all vertices, then train non-vertex Bézier controls with STCH.
4. After the final task, run noisy expected-improvement search over the simplex and rerank the top candidates with repeated evaluation.

The default configuration uses a degree-3 simplex, 30 soft-prompt tokens, `rebase_lr=0.01`, global batch size 128, and eight GPUs.

## Setup

```bash
conda create -n laps python=3.11 -y
conda activate laps
pip install -r requirements.txt
bash scripts/download_trace.sh
```

The expected data directory is `data/TRACE-Benchmark/LLM-CL-Benchmark_5000`.

Build the deterministic 1% replay buffers:

```bash
DATA_ROOT="$PWD/data/TRACE-Benchmark/LLM-CL-Benchmark_5000" \
bash scripts/prepare_replay_buffers.sh
```

## Run

```bash
bash scripts/run_laps.sh \
  --model Qwen/Qwen3-1.7B \
  --data-root "$PWD/data/TRACE-Benchmark/LLM-CL-Benchmark_5000" \
  --gold-root "$PWD/replay_buffers/gold_rho_0.01" \
  --output-dir "$PWD/output/laps_qwen3_1.7b" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --bo-gpu-ids 0,1,2,3,4,5,6,7
```

Use `--diagonal-only-eval` to evaluate only the newly acquired task at each stage. Use `--task-order` with eight comma-separated task names to run a different stream order; build matching replay buffers with the same `TASK_ORDER` value first.

The launcher is resumable. `STAGE_COMPLETE`, `EVAL_COMPLETE`, `FULL_STREAM_COMPLETE`, `SEARCH_COMPLETE`, and `PIPELINE_COMPLETE` mark completed work.

## Layout

```text
scripts/run_laps.sh                         end-to-end entry point
scripts/run_laps_full_stream.sh             resumable continual pipeline
experiments/train_joint_slow_softprompt_vertices.py
experiments/train_trace_simplex_stch_stage.py
experiments/search_trace_simplex_bo_ei_vllm.py
src/simplex_bezier_prompt.py                Bézier simplex parameterization
```

## Citation

A citation will be added after publication.
