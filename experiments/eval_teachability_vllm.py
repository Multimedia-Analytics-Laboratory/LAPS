#!/usr/bin/env python3
"""Evaluate absolute task skill and demonstration-conditioned teachability."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
from pathlib import Path
from string import Template

# FlashInfer's sampler JIT-compiles a kernel on first use and therefore needs
# ninja.  The PyTorch sampler is deterministic here and avoids that unrelated
# runtime dependency without changing greedy decoding semantics.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from trace_task_protocol import ensure_task_prompt, privileged_prompt
from trace_repository_metrics import repository_score


_CANONICAL_TASKS = (
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
)
TASKS = tuple(filter(None, os.environ.get(
    "TRACE_TASK_ORDER", ",".join(_CANONICAL_TASKS),
).split(",")))
if len(TASKS) != 8 or set(TASKS) != set(_CANONICAL_TASKS):
    raise ValueError(f"invalid TRACE_TASK_ORDER: {TASKS}")
MAX_TOKENS = {
    # Demonstration-conditioned prompts explicitly ask for reasoning, so a
    # 1--4 token budget would measure truncation rather than teachability.
    "C-STANCE": 512, "FOMC": 512, "MeetingBank": 512, "Py150": 256,
    "ScienceQA": 512, "NumGLUE-cm": 128, "NumGLUE-ds": 128,
    "20Minuten": 512,
}
TEACHER_TEMPLATE = Template(
    "$question\n\n"
    "This is an example for a response to the question:\n"
    "$answer\n\n"
    "Now answer with a response of your own, including the thinking process."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument("--conditions", nargs="+", choices=("plain", "correct_demo", "shuffled_demo"), required=True)
    parser.add_argument("--probe-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sampling-seed", type=int, default=2026)
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Override the task-specific generation budget.",
    )
    return parser.parse_args()


def first_choice(text: str) -> str:
    # Qwen commonly emits Markdown such as ``**Answer: C. neutral**`` or
    # Chinese ``最终答案是： **B. 反对**``.  Prefer an explicit answer marker;
    # otherwise use the last standalone choice rather than a choice mentioned
    # while reasoning.
    marked = re.findall(
        r"(?is)(?:final\s+answer|answer|最终答案|答案)\s*(?:is|是)?\s*[:：]?\s*\**\s*([A-D])(?=\s|[.):,，、*]|$)",
        text,
    )
    if marked:
        return marked[-1].upper()
    standalone = re.findall(r"(?im)^\s*(?:[-*#>]\s*)*\**\s*([A-D])(?=\s|[.):,，、*]|$)", text)
    return standalone[-1].upper() if standalone else ""


def numeric_answer(text: str) -> str:
    candidates = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return candidates[-1].replace(",", "") if candidates else ""


def lcs_f1(prediction: str, target: str) -> float:
    p, t = prediction.split(), target.split()
    if not p or not t:
        return 0.0
    previous = [0] * (len(t) + 1)
    for a in p:
        current = [0]
        for j, b in enumerate(t, 1):
            current.append(previous[j - 1] + 1 if a == b else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    return 200.0 * lcs / (len(p) + len(t)) if lcs else 0.0


def sari_score(source: str, prediction: str, target: str) -> float:
    source_tokens, pred_tokens, target_tokens = source.lower().split(), prediction.lower().split(), target.lower().split()
    scores = []
    for n in range(1, 5):
        grams = lambda xs: {tuple(xs[i:i+n]) for i in range(max(0, len(xs)-n+1))}
        s, p, t = grams(source_tokens), grams(pred_tokens), grams(target_tokens)
        add_p, add_t = p-s, t-s
        keep_p, keep_t = p&s, t&s
        del_p, del_t = s-p, s-t
        f1 = lambda a, b: (2*len(a&b)/(len(a)+len(b))) if a or b else 1.0
        delete_precision = len(del_p&del_t)/len(del_p) if del_p else (1.0 if not del_t else 0.0)
        scores.append((f1(add_p, add_t)+f1(keep_p, keep_t)+delete_precision)/3)
    return 100.0 * sum(scores) / len(scores)


def score(task: str, predictions: list[str], rows: list[dict]) -> float:
    targets = [str(r["answer"]).strip() for r in rows]
    if task in {"C-STANCE", "FOMC", "ScienceQA"}:
        return 100.0 * sum(first_choice(p) == first_choice(t) for p, t in zip(predictions, targets)) / len(rows)
    if task in {"NumGLUE-cm", "NumGLUE-ds"}:
        return 100.0 * sum(numeric_answer(p) == numeric_answer(t) for p, t in zip(predictions, targets)) / len(rows)
    if task == "Py150":
        return sum(100.0 * difflib.SequenceMatcher(None, p, t).ratio() for p, t in zip(predictions, targets)) / len(rows)
    if task == "MeetingBank":
        return sum(lcs_f1(p, t) for p, t in zip(predictions, targets)) / len(rows)
    return sum(sari_score(str(r["prompt"]), p, t) for r, p, t in zip(rows, predictions, targets)) / len(rows)


def choose_probe(rows: list[dict], tokenizer, task: str, n: int, seed: int, max_tokens: int) -> list[dict]:
    import random
    rng = random.Random(f"{seed}:{task}")
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = []
    for i in indices:
        row = rows[i]
        correct = privileged_prompt(task, str(row["prompt"]), str(row["answer"]))
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": correct}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        if len(tokenizer(rendered, add_special_tokens=False).input_ids) <= max_tokens:
            selected.append({**row, "source_index": i})
        if len(selected) == n:
            break
    return selected


def mismatched_answers(rows: list[dict], task: str) -> list[str]:
    answers = [str(r["answer"]) for r in rows]
    result = []
    for i, answer in enumerate(answers):
        candidate = answers[(i + 1) % len(answers)]
        for offset in range(1, len(answers)):
            candidate = answers[(i + offset) % len(answers)]
            if task not in {"C-STANCE", "FOMC", "ScienceQA"} or first_choice(candidate) != first_choice(answer):
                if candidate != answer:
                    break
        result.append(candidate)
    return result


def render(tokenizer, content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_prompt_tokens + max(MAX_TOKENS.values()),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=128,
        enable_prefix_caching=True,
    )
    result = {"model": args.model, "probe_size_requested": args.probe_size, "tasks": {}}
    for task in args.tasks:
        source = json.loads((args.data_root / task / "test.json").read_text())
        rows = choose_probe(source, tokenizer, task, args.probe_size, args.seed, args.max_prompt_tokens)
        wrong = mismatched_answers(rows, task)
        task_result = {"num_samples": len(rows), "source_indices": [r["source_index"] for r in rows]}
        for condition in args.conditions:
            if condition == "plain":
                contents = [ensure_task_prompt(task, str(r["prompt"])) for r in rows]
            elif condition == "correct_demo":
                contents = [privileged_prompt(task, str(r["prompt"]), str(r["answer"])) for r in rows]
            else:
                contents = [privileged_prompt(task, str(r["prompt"]), w) for r, w in zip(rows, wrong)]
            prompts = [render(tokenizer, x) for x in contents]
            generation_budget = args.max_tokens or MAX_TOKENS[task]
            params = SamplingParams(
                temperature=args.temperature,
                max_tokens=generation_budget,
                n=args.num_samples,
                seed=args.sampling_seed,
            )
            outputs = llm.generate(prompts, params, use_tqdm=False)
            predictions_by_sample = [
                [output.outputs[sample].text.strip() for output in outputs]
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
            task_result[condition] = {
                "score": score_mean,
                "scores": scores,
                "score_std": score_std,
                "temperature": args.temperature,
                "num_samples": args.num_samples,
                "sampling_seed": args.sampling_seed,
                "max_tokens": generation_budget,
                "mean_output_tokens": sum(
                    len(sample.token_ids)
                    for output in outputs for sample in output.outputs
                ) / (len(outputs) * args.num_samples),
                "predictions": predictions_by_sample[0],
                "predictions_by_sample": predictions_by_sample,
            }
        if "plain" in task_result and "correct_demo" in task_result:
            task_result["icl_lift"] = task_result["correct_demo"]["score"] - task_result["plain"]["score"]
        if "correct_demo" in task_result and "shuffled_demo" in task_result:
            task_result["demo_specificity"] = task_result["correct_demo"]["score"] - task_result["shuffled_demo"]["score"]
        result["tasks"][task] = task_result
        compact = {"num_samples": task_result["num_samples"]}
        for condition in args.conditions:
            if args.num_samples == 1:
                compact[condition] = task_result[condition]["score"]
            else:
                compact[condition] = {
                    "mean": task_result[condition]["score"],
                    "std": task_result[condition]["score_std"],
                    "runs": task_result[condition]["scores"],
                }
        for key in ("icl_lift", "demo_specificity"):
            if key in task_result:
                compact[key] = task_result[key]
        print("TEACHABILITY " + json.dumps({task: compact}, ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
