#!/usr/bin/env python3
"""Evaluate one fixed simplex preference with stochastic vLLM decoding."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from eval_trace_preference_vllm import load_embedding, make_inputs
from search_trace_simplex_bo_ei_vllm import actual_prompt_probe, task_max_tokens
from src.simplex_bezier_prompt import SimplexBezierPrompt
from trace_repository_metrics import repository_score

EVAL_TEXT_PROMPT_LIMIT = 2048
EVAL_MAX_GENERATION = 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--preference", nargs="+", type=float)
    parser.add_argument("--slow-only", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-num-seqs", type=int, default=192)
    parser.add_argument(
        "--generation-chunk-size",
        type=int,
        default=32,
        help=(
            "Submit this many evaluation rows to vLLM per generate call. "
            "Chunking avoids a long CPU-only request-ingestion phase for "
            "request-specific prompt embeddings."
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Request-specific prompt embeddings are materialized on CPU before vLLM
    # can schedule GPU work.  Letting every concurrent evaluator inherit all
    # host cores makes the processes contend for memory bandwidth and leaves
    # the GPUs starved.  Match the bounded threading used by the other TRACE
    # preference evaluator.
    cpu_threads = int(os.environ.get("PREFERENCE_EVAL_CPU_THREADS", "8"))
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(min(4, cpu_threads))

    config = json.loads((args.checkpoint / "simplex_config.json").read_text())
    dim = int(config["stage"]) + 1
    if args.slow_only:
        preference = None
    else:
        if args.preference is None:
            raise ValueError("--preference is required unless --slow-only is set")
        preference = torch.tensor(args.preference, dtype=torch.float32)
        if preference.numel() != dim or torch.any(preference < 0):
            raise ValueError(f"preference must contain {dim} nonnegative values")
        preference /= preference.sum()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    memory = None
    if not args.slow_only:
        embedding = load_embedding(args.checkpoint)
        state = torch.load(
            args.checkpoint / "simplex_soft_prompt.pt",
            map_location="cpu", weights_only=True,
        )
        prompt_model = SimplexBezierPrompt(
            dim, int(config["degree"]), int(config["prompt_length"]),
            int(config["hidden_size"]),
            residual_scale=float(config.get("residual_scale", 1.0)),
        )
        prompt_model.load_state_dict(state)
        prompt_model.eval()
        with torch.no_grad():
            memory = prompt_model(preference.unsqueeze(0))[0].to(torch.bfloat16)

    max_tokens = task_max_tokens(args.task)
    eval_max_model_len = (
        EVAL_TEXT_PROMPT_LIMIT + int(config["prompt_length"])
        + EVAL_MAX_GENERATION
    )
    source = json.loads((args.data_root / args.task / "test.json").read_text())
    rows = [
        {**row, "task": args.task}
        for row in actual_prompt_probe(
            source, tokenizer, args.task, 100000, args.seed, 2048,
        )
    ]
    if args.slow_only:
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ) for row in rows]
    else:
        prompts = make_inputs(
            tokenizer, embedding, memory, rows,
            eval_max_model_len - max_tokens - int(config["prompt_length"]),
            "chat_start",
        )
    llm = LLM(
        model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=eval_max_model_len, max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False, enable_prompt_embeds=True,
        enforce_eager=True,
    )
    if args.generation_chunk_size <= 0:
        raise ValueError("--generation-chunk-size must be positive")
    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=max_tokens,
        n=args.num_samples, seed=args.seed,
    )
    outputs = []
    for start in range(0, len(prompts), args.generation_chunk_size):
        stop = min(start + args.generation_chunk_size, len(prompts))
        outputs.extend(llm.generate(
            prompts[start:stop], sampling_params, use_tqdm=False,
        ))
        print(json.dumps({
            "event": "simplex_point_eval_progress",
            "task": args.task,
            "completed_rows": stop,
            "total_rows": len(prompts),
            "generation_chunk_size": args.generation_chunk_size,
        }), flush=True)
    predictions = [
        [item.outputs[i].text.strip() for item in outputs]
        for i in range(args.num_samples)
    ]
    scores = [repository_score(args.task, values, rows) for values in predictions]
    mean = float(sum(scores) / len(scores))
    std = float(math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores)))
    result = {
        "checkpoint": str(args.checkpoint), "task": args.task,
        "preference": None if preference is None else preference.tolist(),
        "slow_only": args.slow_only, "num_test_rows": len(rows),
        "temperature": args.temperature, "num_samples": args.num_samples,
        "scores": scores, "score": mean, "score_std": std,
        "mean_output_tokens": sum(
            len(sample.token_ids) for item in outputs for sample in item.outputs
        ) / (len(rows) * args.num_samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "simplex_point_eval", **result}), flush=True)
    engine_core = getattr(llm.llm_engine, "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        engine_core.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
