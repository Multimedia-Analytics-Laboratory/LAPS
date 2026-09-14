#!/usr/bin/env python3
"""Greedy TRACE evaluation of a preference-conditioned checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.preference_memory import prompt_position  # noqa: E402
from src.input_conditioned_softprompt import InputConditionedBezierCode  # noqa: E402
from src.task_conditioned_softprompt import TaskConditionedBezierCode  # noqa: E402
from eval_teachability_vllm import TASKS, choose_probe  # noqa: E402
from trace_paper_metrics import paper_score  # noqa: E402
from trace_repository_metrics import repository_score  # noqa: E402
from trace_task_protocol import ensure_task_prompt  # noqa: E402

EVAL_TEXT_PROMPT_LIMIT = 2048
EVAL_MAX_GENERATION = 512


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    p.add_argument("--lambdas", nargs="+", type=float, default=(0, 0.5, 1))
    p.add_argument("--task-lambda-map", type=Path)
    p.add_argument("--probe-size", type=int, default=500)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument(
        "--gpu-memory-utilization", type=float, default=0.82,
        help="Fraction of GPU memory reserved by vLLM (lower for input-conditioned evaluators).",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--sampling-seed", type=int, default=2026)
    p.add_argument(
        "--prompt-placement", choices=("auto", "end", "before_assistant", "chat_start"),
        default="auto",
        help=(
            "Where to insert fast-memory tokens. auto reads the placement "
            "recorded by training and prevents train/eval coordinate drift."
        ),
    )
    p.add_argument(
        "--slow-only", action="store_true",
        help="Evaluate the backbone without loading or inserting fast-memory tokens.",
    )
    p.add_argument(
        "--include-slow-only", action="store_true",
        help="Evaluate slow-only in addition to all requested preference values.",
    )
    return p.parse_args()


def load_embedding(checkpoint):
    name = "model.embed_tokens.weight"
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        paths = [checkpoint / index["weight_map"][name]]
    else:
        paths = sorted(checkpoint.glob("*.safetensors"))
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name).clone().to(torch.bfloat16)
    raise KeyError(name)


def memory_at(controls, lam):
    order = int(controls.shape[0]) - 1
    basis = torch.tensor([
        math.comb(order, index)
        * (1 - lam) ** (order - index)
        * lam ** index
        for index in range(order + 1)
    ])
    return torch.einsum("k,kmh->mh", basis, controls.float()).to(torch.bfloat16)


def make_inputs(tokenizer, embedding, memory, rows, max_prompt_tokens,
                prompt_placement="end"):
    result = []
    for row_index, row in enumerate(rows):
        task = str(row.get("task", ""))
        content = ensure_task_prompt(task, str(row["prompt"])) if task else str(row["prompt"])
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer(
            text, add_special_tokens=False, truncation=True,
            max_length=max_prompt_tokens,
        ).input_ids
        position = (
            len(ids) if prompt_placement == "end" else
            0 if prompt_placement == "chat_start" else
            prompt_position(tokenizer, ids)
        )
        values = embedding[torch.tensor(ids, dtype=torch.long)]
        row_memory = memory[row_index] if memory.ndim == 3 else memory
        result.append({"prompt_embeds": torch.cat((
            values[:position], row_memory, values[position:],
        )).contiguous()})
    return result


def make_plain_inputs(tokenizer, rows, max_prompt_tokens):
    result = []
    for row in rows:
        task = str(row.get("task", ""))
        content = ensure_task_prompt(task, str(row["prompt"])) if task else str(row["prompt"])
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer(
            text, add_special_tokens=False, truncation=True,
            max_length=max_prompt_tokens,
        ).input_ids
        result.append(tokenizer.decode(ids, skip_special_tokens=False))
    return result


def main():
    args = parse_args()
    # Input-conditioned evaluation runs the frozen question encoder on CPU.
    # A single thread makes full TRACE test sets needlessly dominate runtime.
    cpu_threads = int(os.environ.get("PREFERENCE_EVAL_CPU_THREADS", "8"))
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(min(4, cpu_threads))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preference_config = json.loads(
        (args.checkpoint / "preference_config.json").read_text()
    )
    configured_placement = preference_config.get(
        "soft_prompt_placement", "chat_start",
    )
    if args.prompt_placement == "auto":
        args.prompt_placement = configured_placement
    elif args.prompt_placement != configured_placement:
        raise ValueError(
            "soft-prompt placement mismatch: checkpoint was trained with "
            f"{configured_placement!r}, evaluation requested "
            f"{args.prompt_placement!r}"
        )
    continual_stage = int(preference_config["stage"])
    requested_tasks = list(args.tasks)
    seen_tasks = set(TASKS[:continual_stage + 1])
    args.tasks = [task for task in requested_tasks if task in seen_tasks]
    result = {
        "checkpoint": str(args.checkpoint),
        "evaluation_scope": "seen tasks through current continual stage",
        "continual_stage": continual_stage,
        "requested_tasks": requested_tasks,
        "evaluated_tasks": list(args.tasks),
        "skipped_unseen_tasks": [
            task for task in requested_tasks if task not in seen_tasks
        ],
        "tasks": {},
    }
    print(json.dumps({
        "event": "eval_scope",
        "continual_stage": continual_stage,
        "requested_tasks": requested_tasks,
        "evaluated_tasks": list(args.tasks),
        "skipped_unseen_tasks": result["skipped_unseen_tasks"],
    }, ensure_ascii=False), flush=True)
    # The four-GPU launcher may assign a shard containing only future tasks.
    # Exit before loading the checkpoint so that such a shard consumes no GPU.
    if not args.tasks:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return
    slow_checkpoint = Path(preference_config.get("slow_checkpoint", args.checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(slow_checkpoint)
    embedding = None if args.slow_only else load_embedding(slow_checkpoint)
    state = None if args.slow_only else torch.load(
        args.checkpoint / "preference_soft_prompt.pt", map_location="cpu",
        weights_only=True,
    )
    input_conditioned = bool(preference_config.get("input_conditioned", False))
    task_conditioned = bool(preference_config.get("task_conditioned", False))
    stage_conditioned_path = bool(
        preference_config.get("stage_conditioned_path", False)
    )
    # The reference evaluator keys the generation budget by the current
    # continual stage (task_id), not by the evaluated task identity.
    eval_max_tokens = 1 if continual_stage in {0, 1} else 512
    controls = None if state is None else state["controls"]
    code_model = None
    if state is not None and input_conditioned:
        code_model = InputConditionedBezierCode(
            int(preference_config["prompt_length"]),
            int(controls.shape[-1]),
            condition_width=int(preference_config.get("condition_width") or 512),
            condition_rank=int(preference_config.get("condition_rank") or 8),
            max_question_length=int(
                preference_config.get("max_question_encoder_length") or 384
            ),
            conditional_gate=bool(
                preference_config.get("condition_acquire_gate", True)
            ),
            bezier_order=int(preference_config.get("bezier_order", 3)),
        )
        code_model.text_encoder.float()
        code_model.load_state_dict(state)
        code_model.eval()
    prompt_length = 0 if controls is None else (
        controls.shape[-2] if task_conditioned else controls.shape[1]
    )
    eval_max_model_len = (
        EVAL_TEXT_PROMPT_LIMIT + prompt_length + EVAL_MAX_GENERATION
    )
    llm = LLM(
        model=str(slow_checkpoint), dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=eval_max_model_len, max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_prompt_embeds=True, enforce_eager=True,
    )
    task_lambda_map = (
        json.loads(args.task_lambda_map.read_text())
        if args.task_lambda_map else None
    )
    for task in args.tasks:
        source = json.loads((args.data_root / task / "test.json").read_text())
        rows = [{**row, "task": task} for row in choose_probe(source, tokenizer, task, args.probe_size, 2026, 2048)]
        question_features = None
        if code_model is not None:
            contents = [
                ensure_task_prompt(task, str(row["prompt"])) for row in rows
            ]
            question_features = code_model.encode_questions(
                contents, torch.device("cpu"),
            )
        task_result = {
            "num_samples": len(rows),
            "source_indices": [row["source_index"] for row in rows],
            "metric_protocol": "TRACE paper task metrics; robust answer extraction",
            "generation_protocol": "stage-conditioned budget",
            "continual_stage": continual_stage,
            "max_tokens": eval_max_tokens,
            "lambdas": {},
        }
        if args.include_slow_only and not args.slow_only:
            plain_outputs = llm.generate(
                make_plain_inputs(
                    tokenizer, rows, eval_max_model_len - eval_max_tokens,
                ),
                SamplingParams(
                    temperature=args.temperature, max_tokens=eval_max_tokens,
                    n=args.num_samples, seed=args.sampling_seed,
                ),
                use_tqdm=False,
            )
            plain_predictions_by_sample = [
                [value.outputs[sample].text.strip() for value in plain_outputs]
                for sample in range(args.num_samples)
            ]
            plain_scores = [
                repository_score(task, predictions, rows)
                for predictions in plain_predictions_by_sample
            ]
            plain_mean = sum(plain_scores) / len(plain_scores)
            plain_std = math.sqrt(sum(
                (score - plain_mean) ** 2 for score in plain_scores
            ) / len(plain_scores))
            task_result["lambdas"]["slow_only"] = {
                "score": plain_mean,
                "scores": plain_scores,
                "score_std": plain_std,
                "temperature": args.temperature,
                "num_samples": args.num_samples,
                "sampling_seed": args.sampling_seed,
                "mean_output_tokens": sum(
                    len(sample.token_ids)
                    for value in plain_outputs for sample in value.outputs
                ) / (len(plain_outputs) * args.num_samples),
                "predictions": plain_predictions_by_sample[0],
                "predictions_by_sample": plain_predictions_by_sample,
            }
            plain_summary = task_result["lambdas"]["slow_only"]
            print(json.dumps({
                "event": "preference_eval", "task": task,
                "lambda": "slow_only", "score": plain_summary["score"],
                "scores": plain_summary["scores"],
                "score_std": plain_summary["score_std"],
                "mean_output_tokens": plain_summary["mean_output_tokens"],
            }), flush=True)
        merged_prompts = []
        task_lambdas = (
            [None] if args.slow_only else
            [float(task_lambda_map[task])]
            if task_lambda_map is not None else args.lambdas
        )
        for lam in task_lambdas:
            if lam is None:
                merged_prompts.extend(make_plain_inputs(
                    tokenizer, rows, eval_max_model_len - eval_max_tokens,
                ))
            else:
                if code_model is None:
                    selected_controls = (
                        controls[
                            continual_stage
                            if stage_conditioned_path else TASKS.index(task)
                        ]
                        if task_conditioned else controls
                    )
                    memory = memory_at(selected_controls, lam)
                else:
                    memory = code_model(
                        torch.full((len(rows),), float(lam)),
                        question_features,
                    ).detach().to(torch.bfloat16)
                merged_prompts.extend(make_inputs(
                    tokenizer, embedding, memory, rows,
                    eval_max_model_len - eval_max_tokens - prompt_length,
                    args.prompt_placement,
                ))
        merged_outputs = llm.generate(
            merged_prompts,
            SamplingParams(
                temperature=args.temperature, max_tokens=eval_max_tokens,
                n=args.num_samples, seed=args.sampling_seed,
            ),
            use_tqdm=False,
        )
        width = len(rows)
        for lambda_index, lam in enumerate(task_lambdas):
            outputs = merged_outputs[
                lambda_index * width:(lambda_index + 1) * width
            ]
            predictions_by_sample = [
                [value.outputs[sample].text.strip() for value in outputs]
                for sample in range(args.num_samples)
            ]
            scores = [
                repository_score(task, predictions, rows)
                for predictions in predictions_by_sample
            ]
            score_mean = sum(scores) / len(scores)
            score_std = math.sqrt(sum(
                (score - score_mean) ** 2 for score in scores
            ) / len(scores))
            key = "slow_only" if lam is None else str(lam)
            task_result["lambdas"][key] = {
                "score": score_mean,
                "scores": scores,
                "score_std": score_std,
                "temperature": args.temperature,
                "num_samples": args.num_samples,
                "sampling_seed": args.sampling_seed,
                "mean_output_tokens": sum(
                    len(sample.token_ids)
                    for value in outputs for sample in value.outputs
                ) / (len(outputs) * args.num_samples),
                "predictions": predictions_by_sample[0],
                "predictions_by_sample": predictions_by_sample,
            }
            summary = task_result["lambdas"][key]
            print(json.dumps({
                "event": "preference_eval", "task": task,
                "lambda": key, "score": summary["score"],
                "scores": summary["scores"],
                "score_std": summary["score_std"],
                "mean_output_tokens": summary["mean_output_tokens"],
            }), flush=True)
        result["tasks"][task] = task_result
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    engine_core = getattr(llm.llm_engine, "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        engine_core.shutdown()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
