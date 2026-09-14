#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PYTHON:-python}
TORCHRUN=${TORCHRUN:-torchrun}
DATA=${DATA_ROOT:?Set DATA_ROOT to TRACE-Benchmark/LLM-CL-Benchmark_5000}
GOLD=${GOLD_ROOT:-$ROOT/replay_buffers/gold_rho_0.01}
RUN=${RUN:-$ROOT/output/joint_slow_softprompt_vertices_cstance_fomc}
START_STAGE=${START_STAGE:-2}
SEED_RUN=${SEED_RUN:-$ROOT/output/joint_slow_softprompt_vertices_cstance_fomc}
INITIAL_MODEL=${INITIAL_MODEL:-}
REPLAY_GLOBAL_BATCH=${REPLAY_GLOBAL_BATCH:-8}
PROMPT_LENGTH=${PROMPT_LENGTH:-30}
BEZIER_DEGREE=${BEZIER_DEGREE:-3}
PROMPT_LR=${PROMPT_LR:-0.3}
SLOW_LR=${SLOW_LR:-1e-5}
REBASE_LR=${REBASE_LR:-0.01}
HISTORICAL_VERTEX_LR_OVERRIDES=${HISTORICAL_VERTEX_LR_OVERRIDES:-}
POST_SLOW_VERTEX_LR=${POST_SLOW_VERTEX_LR:-0.03}
PROMOT_PER_DEVICE_BATCH=${PROMOT_PER_DEVICE_BATCH:-4}
SLOW_REPLAY_SFT=${SLOW_REPLAY_SFT:-0}
EVAL_INTERMEDIATE_STAGES=${EVAL_INTERMEDIATE_STAGES:-1}
EVAL_SLOW_BASELINE=${EVAL_SLOW_BASELINE:-1}
EVAL_DIAGONAL_ONLY=${EVAL_DIAGONAL_ONLY:-0}
BO_REUSE_VERTEX_TEST_RESULTS=${BO_REUSE_VERTEX_TEST_RESULTS:-1}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
BO_GPU_IDS=${BO_GPU_IDS:-4,5,6,7}
BO_SCRIPT=${BO_SCRIPT:-$ROOT/scripts/run_simplex_bo.sh}
BO_OUT_NAME=${BO_OUT_NAME:-simplex_bo_noisy_ei}
TASK_ORDER=${TASK_ORDER:-C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten}
REBASE_ENABLED=$(awk -v lr="$REBASE_LR" 'BEGIN { print ((lr + 0) != 0) ? 1 : 0 }')
if [[ -n "$HISTORICAL_VERTEX_LR_OVERRIDES" ]]; then
  REBASE_ENABLED=1
fi
MASTER_LOG=$RUN/logs/full_stream.log
IFS=',' read -r -a TASKS <<< "$TASK_ORDER"
if [[ ${#TASKS[@]} -ne 8 ]]; then
  echo "TASK_ORDER must contain exactly eight comma-separated tasks" >&2
  exit 2
fi
export TRACE_TASK_ORDER=$TASK_ORDER

DATA=$(realpath "$DATA")
GOLD=$(realpath "$GOLD")
RUN=$(realpath -m "$RUN")
SEED_RUN=$(realpath -m "$SEED_RUN")

declare -A TASK_EPOCHS=(
  [C-STANCE]=5 [FOMC]=3 [MeetingBank]=7 [Py150]=5
  [ScienceQA]=3 [NumGLUE-cm]=5 [NumGLUE-ds]=5 [20Minuten]=7
)

mkdir -p "$RUN/logs"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

event() {
  printf '%s event=%s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$MASTER_LOG"
}

run_logged() {
  local name=$1; shift
  event "START name=$name"
  set +e
  "$@" >"$RUN/logs/$name.log" 2>&1
  local rc=$?
  set -e
  event "END name=$name exit_code=$rc"
  return "$rc"
}

evaluate_before_fast() {
  local stage=$1 checkpoint=$2 task_count=$((stage + 1))
  local eval_dir="$checkpoint/eval_before_fast_temp01_x8"
  mkdir -p "$eval_dir"
  event "EVAL_BEFORE_FAST_START stage=$stage checkpoint=$checkpoint"
  local failed=0
  local total_jobs=$task_count
  local gpu_jobs=8
  if [[ "$EVAL_DIAGONAL_ONLY" == "1" ]]; then
    total_jobs=1
    gpu_jobs=1
  elif [[ "$EVAL_SLOW_BASELINE" == "1" ]]; then
    total_jobs=$((2 * task_count))
  fi
  local pids=()
  # Give each GPU a persistent sequential queue.  The previous global
  # wait-any scheduler could launch job 9 on GPU 0 merely because *another*
  # GPU finished first, colliding with GPU 0's still-live vLLM KV cache.
  for ((gpu=0; gpu<gpu_jobs; gpu++)); do
    (
      cd /tmp
      # vLLM V0 asks the OS for an internal c10d port during engine startup.
      # Starting eight engines in the same millisecond can race on that free
      # port even though each outer evaluation has a distinct MASTER_PORT.
      sleep $((gpu * 2))
      for ((job=gpu; job<total_jobs; job+=8)); do
        if [[ "$EVAL_DIAGONAL_ONLY" == "1" ]]; then
          mode=vertex
          task_id=$stage
        elif ((job < task_count)); then
          mode=vertex
          task_id=$job
        else
          mode=slow_only
          task_id=$((job - task_count))
        fi
        task=${TASKS[$task_id]}
        port=$((41000 + stage * 100 + job))
        output="$eval_dir/${task}_${mode}.json"
        [[ -s "$output" ]] && continue
        preference=()
        for ((j=0; j<task_count; j++)); do
          if ((j == task_id)); then preference+=(1); else preference+=(0); fi
        done
        rc=1
        for attempt in 0 1 2; do
          attempt_port=$((port + attempt * 1000))
          log_mode=$mode
          [[ "$mode" == slow_only ]] && log_mode=slow
          log="$RUN/logs/eval_stage$(printf '%02d' "$stage")_${task}_${log_mode}.log"
          if [[ "$mode" == vertex ]]; then
            if CUDA_VISIBLE_DEVICES=$gpu PYTHONSAFEPATH=1 \
              MASTER_ADDR=127.0.0.1 MASTER_PORT=$attempt_port "$PY" -P \
              "$ROOT/experiments/eval_trace_simplex_point_vllm.py" \
              --checkpoint "$checkpoint" --data-root "$DATA" --task "$task" \
              --preference "${preference[@]}" --temperature 0.1 --num-samples 8 \
              --output "$output" >"$log" 2>&1; then rc=0; break; fi
          else
            if CUDA_VISIBLE_DEVICES=$gpu PYTHONSAFEPATH=1 \
              MASTER_ADDR=127.0.0.1 MASTER_PORT=$attempt_port "$PY" -P \
              "$ROOT/experiments/eval_trace_simplex_point_vllm.py" \
              --checkpoint "$checkpoint" --data-root "$DATA" --task "$task" \
              --slow-only --temperature 0.1 --num-samples 8 \
              --output "$output" >"$log" 2>&1; then rc=0; break; fi
          fi
          rm -f "$output"
          sleep $((5 * (attempt + 1)))
        done
        ((rc == 0)) || exit "$rc"
      done
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if ((failed)); then
    event "EVAL_BEFORE_FAST_FAILED stage=$stage"
    return 1
  fi
  touch "$eval_dir/EVAL_COMPLETE"
  if [[ "$EVAL_DIAGONAL_ONLY" == "1" ]]; then
    touch "$eval_dir/DIAGONAL_EVAL_COMPLETE"
  fi
  event "EVAL_BEFORE_FAST_END stage=$stage"
}

event "CONFIG task_order=$TASK_ORDER slow_replay_sft=$SLOW_REPLAY_SFT replay_global_batch=$REPLAY_GLOBAL_BATCH prompt_length=$PROMPT_LENGTH bezier_degree=$BEZIER_DEGREE prompt_lr=$PROMPT_LR slow_lr=$SLOW_LR rebase_lr=$REBASE_LR rebase_enabled=$REBASE_ENABLED historical_vertex_lr_overrides=$HISTORICAL_VERTEX_LR_OVERRIDES post_slow_vertex_lr=$POST_SLOW_VERTEX_LR promot_per_device_batch=$PROMOT_PER_DEVICE_BATCH eval_intermediate_stages=$EVAL_INTERMEDIATE_STAGES eval_slow_baseline=$EVAL_SLOW_BASELINE eval_diagonal_only=$EVAL_DIAGONAL_ONLY bo_reuse_vertex_test_results=$BO_REUSE_VERTEX_TEST_RESULTS gpu_ids=$GPU_IDS bo_gpu_ids=$BO_GPU_IDS"

# A continuation may reuse the last valid stage from another run.  In the
# multi-token repair, START_STAGE=3 reuses the completed MeetingBank stage and
# reruns exactly the first stage whose historical set contains a generation
# task.
if ((START_STAGE == 0)); then
  if [[ -z "$INITIAL_MODEL" ]]; then
    echo "INITIAL_MODEL is required when START_STAGE=0" >&2
    exit 2
  fi
  PREV_SLOW=$(realpath "$INITIAL_MODEL")
  PREV_PROMPT=
else
  seed_stage=$((START_STAGE - 1))
  seed_task=${TASKS[$seed_stage]}
  PREV_SLOW=${SEED_SLOW:-$SEED_RUN/$(printf 'stage_%02d_%s' "$seed_stage" "$seed_task")}
  PREV_PROMPT=${SEED_PROMPT:-$SEED_RUN/$(printf 'stage_%02d_%s_fast/simplex_soft_prompt.pt' "$seed_stage" "$seed_task")}
  PREV_SLOW=$(realpath -m "$PREV_SLOW")
  PREV_PROMPT=$(realpath -m "$PREV_PROMPT")
  if [[ ! -f "$PREV_PROMPT" ]]; then
    echo "Missing completed seed Fast prompt: $PREV_PROMPT" >&2
    exit 2
  fi
fi

for ((stage=START_STAGE; stage<=7; stage++)); do
  task=${TASKS[$stage]}
  buffer=
  if ((stage > 0)); then
    previous=$((stage - 1))
    previous_task=${TASKS[$previous]}
    buffer="$GOLD/$(printf 'stage_%02d_%s' "$previous" "$previous_task")/buffer.jsonl"
  fi
  vertex_out="$RUN/$(printf 'stage_%02d_%s' "$stage" "$task")"
  fast_out="${vertex_out}_fast"
  promot_raw="${vertex_out}_promot_raw"
  promot_prompt="$RUN/promot_task_prompts/$(printf 'stage_%02d_%s.pt' "$stage" "$task")"

  if [[ ! -f "$vertex_out/STAGE_COMPLETE" ]]; then
    mkdir -p "$promot_raw" "$(dirname "$promot_prompt")"
    common_promot=(
      --task "$task" --stage "$stage" --model "$PREV_SLOW"
      --data-root "$DATA" --output-dir "$promot_raw" --prompt-file "$promot_prompt"
      --max-train-samples 5000 --epochs "${TASK_EPOCHS[$task]}"
      --model-learning-rate "$SLOW_LR" --prompt-learning-rate "$PROMPT_LR"
      --prompt-length "$PROMPT_LENGTH" --max-length 2048 --seed 42
    )
    if [[ ! -f "$promot_raw/PROMPT_COMPLETE" ]]; then
      promot_prompt_succeeded=0
      for batch in 16 8 4 2 1; do
        ((batch > PROMOT_PER_DEVICE_BATCH)) && continue
        accum=$((128 / (8 * batch)))
        attempt_name="promot_prompt_$(printf '%02d' "$stage")_${task}_b${batch}"
        if run_logged "$attempt_name" \
            "$PY" -m accelerate.commands.launch --num_processes 8 \
            --main_process_port $((52000 + stage * 4)) \
            "$ROOT/experiments/train_trace_promot_phase.py" --phase prompt \
            "${common_promot[@]}" --per-device-batch-size "$batch" \
            --gradient-accumulation-steps "$accum"; then
          promot_prompt_succeeded=1
          break
        fi
        if ! rg -q 'CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError' \
            "$RUN/logs/$attempt_name.log"; then
          event "PROMOT_PROMPT_FAILED_NON_OOM stage=$stage task=$task batch=$batch"
          exit 1
        fi
        event "PROMOT_PROMPT_OOM stage=$stage task=$task batch=$batch fallback=next"
        sleep 10
      done
      if (( ! promot_prompt_succeeded )); then
        event "PROMOT_PROMPT_OOM_EXHAUSTED stage=$stage task=$task"
        exit 1
      fi
    fi
    if [[ ! -f "$promot_raw/MODEL_COMPLETE" ]]; then
      model_replay_args=()
      if ((stage > 0)) && [[ "$SLOW_REPLAY_SFT" == "1" ]]; then
        model_replay_args+=(--slow-replay-sft --buffer "$buffer")
      fi
      promot_slow_succeeded=0
      for batch in 16 8 4 2 1; do
        ((batch > PROMOT_PER_DEVICE_BATCH)) && continue
        accum=$((128 / (8 * batch)))
        attempt_name="promot_slow_$(printf '%02d' "$stage")_${task}_b${batch}"
        if run_logged "$attempt_name" \
            "$PY" -m accelerate.commands.launch --num_processes 8 --use_deepspeed \
            --zero_stage 1 --gradient_accumulation_steps "$accum" --gradient_clipping 1.0 \
            --main_process_port $((52001 + stage * 4)) \
            "$ROOT/experiments/train_trace_promot_phase.py" --phase model \
            "${common_promot[@]}" --per-device-batch-size "$batch" \
            --gradient-accumulation-steps "$accum" "${model_replay_args[@]}"; then
          promot_slow_succeeded=1
          break
        fi
        if ! rg -q 'CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError' \
            "$RUN/logs/$attempt_name.log"; then
          event "PROMOT_SLOW_FAILED_NON_OOM stage=$stage task=$task batch=$batch"
          exit 1
        fi
        event "PROMOT_SLOW_OOM stage=$stage task=$task batch=$batch fallback=next"
        sleep 10
      done
      if (( ! promot_slow_succeeded )); then
        event "PROMOT_SLOW_OOM_EXHAUSTED stage=$stage task=$task"
        exit 1
      fi
    fi

    assembled_prompt="$promot_raw/simplex_soft_prompt.pt"
    merge_args=(
      "$ROOT/experiments/merge_promot_vertex_into_simplex.py"
      --stage "$stage" --degree "$BEZIER_DEGREE" --promot-prompt "$promot_prompt"
      --output "$assembled_prompt"
    )
    ((stage > 0)) && merge_args+=(--previous-prompt "$PREV_PROMPT")
    "$PY" "${merge_args[@]}"

    prompt_steps=$(( (5000 + 127) / 128 * TASK_EPOCHS[$task] ))
    if [[ -s "$promot_raw/prompt_metrics.json" ]]; then
      rows=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"])' "$promot_raw/prompt_metrics.json")
      prompt_steps=$(( (rows + 127) / 128 * TASK_EPOCHS[$task] ))
    fi
    slow_steps=$prompt_steps
    if [[ -s "$promot_raw/model_metrics.json" ]]; then
      rows=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"])' "$promot_raw/model_metrics.json")
      slow_steps=$(( (rows + 127) / 128 * TASK_EPOCHS[$task] ))
    fi
    post_slow_vertex_steps=$(( (prompt_steps + 9) / 10 ))
    rebase_steps=0
    if ((stage > 0 && REBASE_ENABLED == 1)); then
      rebase_steps=$(( (slow_steps + 1) / 2 ))
    fi
    rebase_args=()
    if ((stage > 0 && REBASE_ENABLED == 1)); then
      rebase_args+=(
        --teacher-model "$PREV_SLOW" --teacher-prompt "$PREV_PROMPT"
        --buffer "$buffer"
      )
    fi
    if [[ -n "$HISTORICAL_VERTEX_LR_OVERRIDES" ]]; then
      read -r -a vertex_lr_overrides <<<"$HISTORICAL_VERTEX_LR_OVERRIDES"
      for specification in "${vertex_lr_overrides[@]}"; do
        rebase_args+=(--historical-vertex-lr "$specification")
      done
    fi
    run_logged "postslow_recalibrate_rebase_$(printf '%02d' "$stage")_${task}" \
      "$TORCHRUN" --standalone --nproc_per_node=8 \
      "$ROOT/experiments/train_joint_slow_softprompt_vertices.py" \
      --stage "$stage" --task "$task" --model "$promot_raw" \
      --previous-prompt "$assembled_prompt" "${rebase_args[@]}" \
      --data-root "$DATA" --output-dir "$vertex_out" --global-batch 128 \
      --replay-global-batch "$REPLAY_GLOBAL_BATCH" --max-train-samples 5000 \
      --prompt-length "$PROMPT_LENGTH" --degree "$BEZIER_DEGREE" --old-prompt-lr "$REBASE_LR" \
      --condition-chunk 2 --transport-only-steps "$rebase_steps" \
      --post-slow-current-vertex-steps "$post_slow_vertex_steps" \
      --post-slow-current-vertex-lr "$POST_SLOW_VERTEX_LR"
  else
    event "SKIP_VERTEX_COMPLETE stage=$stage task=$task"
  fi

  eval_dir="$vertex_out/eval_before_fast_temp01_x8"
  if [[ "$EVAL_DIAGONAL_ONLY" == "1" ]] && \
      [[ ! -f "$eval_dir/DIAGONAL_EVAL_COMPLETE" ]]; then
    evaluate_before_fast "$stage" "$vertex_out"
  elif [[ "$EVAL_DIAGONAL_ONLY" == "1" ]]; then
    event "SKIP_DIAGONAL_EVAL_COMPLETE stage=$stage task=$task"
  elif ((stage < 7)) && [[ "$EVAL_INTERMEDIATE_STAGES" != "1" ]]; then
    event "SKIP_INTERMEDIATE_EVAL stage=$stage task=$task"
  elif [[ ! -f "$eval_dir/EVAL_COMPLETE" ]]; then
    evaluate_before_fast "$stage" "$vertex_out"
  else
    event "SKIP_EVAL_COMPLETE stage=$stage task=$task"
  fi

  if ((stage == 0)); then
    # A one-task simplex contains only its single vertex, so there are no
    # non-vertex controls for the Fast/STCH phase.
    event "SKIP_FAST_NO_INTERIOR stage=0 task=$task"
  elif [[ ! -f "$fast_out/STAGE_COMPLETE" ]]; then
    fast_name="train_fast_$(printf '%02d' "$stage")_${task}"
    fast_cmd=(
      "$TORCHRUN" --standalone --nproc_per_node=8
      "$ROOT/experiments/train_trace_simplex_stch_stage.py"
      --task "$task" --stage "$stage" --model "$vertex_out"
      --data-root "$DATA" --buffer "$buffer"
      --previous-prompt "$vertex_out/simplex_soft_prompt.pt"
      --load-same-stage-prompt --fast-only --train-all-fast-controls --freeze-all-vertices
      --preserve-objective sft --replay-mode per_task_one
      --output-dir "$fast_out" --max-train-samples 5000
      --epochs "${TASK_EPOCHS[$task]}" --global-batch 128
      --prompt-length "$PROMPT_LENGTH" --bezier-degree "$BEZIER_DEGREE" --fast-lr 0.3
      --warmup-steps 10 --lr-scheduler-type linear --stch-mu 0.1
      --disable-stch-normalization --preference-count 8
    )
    fast_succeeded=0
    # Exact streaming fallbacks.  The number of preferences and the global
    # sample batch never change; only the number of retained activation graphs
    # per forward/backward call is reduced.
    for fast_chunks in "8 4" "8 2" "4 2" "2 2" "1 1"; do
      read -r fast_preference_chunk fast_condition_chunk <<<"$fast_chunks"
      attempt_name="${fast_name}_p${fast_preference_chunk}_c${fast_condition_chunk}"
      attempt_cmd=(
        "${fast_cmd[@]}"
        --preference-chunk "$fast_preference_chunk"
        --condition-chunk "$fast_condition_chunk"
        --gradient-checkpointing
      )
      if run_logged "$attempt_name" "${attempt_cmd[@]}"; then
        fast_succeeded=1
        break
      fi

      if ! rg -q 'CUDA out of memory|torch.OutOfMemoryError' \
          "$RUN/logs/$attempt_name.log"; then
        event "FAST_FAILED_NON_OOM stage=$stage task=$task preference_chunk=$fast_preference_chunk condition_chunk=$fast_condition_chunk"
        exit 1
      fi

      event "FAST_OOM stage=$stage task=$task preference_chunk=$fast_preference_chunk condition_chunk=$fast_condition_chunk"
      if [[ -d "$fast_out" ]]; then
        mv "$fast_out" \
          "${fast_out}.oom_p${fast_preference_chunk}_c${fast_condition_chunk}_$(date +%Y%m%d_%H%M%S)"
      fi
    done
    if (( ! fast_succeeded )); then
      event "FAST_OOM_EXHAUSTED stage=$stage task=$task tried=p8c4,p8c2,p4c2,p2c2,p1c1 checkpointing=on"
      exit 1
    fi
  else
    event "SKIP_FAST_COMPLETE stage=$stage task=$task"
  fi

  PREV_SLOW="$vertex_out"
  if ((stage == 0)); then
    PREV_PROMPT="$vertex_out/simplex_soft_prompt.pt"
  else
    PREV_PROMPT="$fast_out/simplex_soft_prompt.pt"
  fi
done

touch "$RUN/FULL_STREAM_COMPLETE"
event "FULL_STREAM_COMPLETE"

# The Fast-stage output is a self-contained checkpoint: it contains the final
# Slow model/tokenizer together with the final simplex prompt.  Run the noisy
# EI preference search from that exact checkpoint so a resumed pipeline cannot
# accidentally evaluate the pre-Fast prompt saved in the vertex directory.
FINAL_TASK=${TASKS[7]}
FINAL_CHECKPOINT="$RUN/stage_07_${FINAL_TASK}_fast"
BO_OUT="$RUN/$BO_OUT_NAME"
if [[ ! -f "$BO_OUT/SEARCH_COMPLETE" ]]; then
  event "BO_START checkpoint=$FINAL_CHECKPOINT output=$BO_OUT"
  run_logged "simplex_bo_noisy_ei" \
    env CHECKPOINT="$FINAL_CHECKPOINT" DATA_ROOT="$DATA" OUT="$BO_OUT" \
        GOLD_ROOT="$GOLD" GPU_IDS="$BO_GPU_IDS" PYTHON="$PY" \
        REUSE_VERTEX_TEST_RESULTS="$BO_REUSE_VERTEX_TEST_RESULTS" \
        TASK_ORDER="$TASK_ORDER" TRACE_TASK_ORDER="$TASK_ORDER" \
    "$BO_SCRIPT"
  event "BO_COMPLETE output=$BO_OUT"
else
  event "SKIP_BO_COMPLETE output=$BO_OUT"
fi

touch "$RUN/PIPELINE_COMPLETE"
event "PIPELINE_COMPLETE"
