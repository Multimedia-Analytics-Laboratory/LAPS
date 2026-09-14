#!/usr/bin/env python3
"""Batched expected-improvement search on a preference simplex."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from src.simplex_bezier_prompt import SimplexBezierPrompt
from eval_teachability_vllm import TASKS, choose_probe
from eval_trace_preference_vllm import load_embedding, make_inputs, make_plain_inputs
from trace_repository_metrics import repository_score, repository_scores_per_example
from trace_task_protocol import ensure_task_prompt

EVAL_TEXT_PROMPT_LIMIT = 2048
EVAL_MAX_GENERATION = 512


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    p.add_argument("--budget", type=int, default=40)
    p.add_argument(
        "--task-budget", action="append", default=[],
        help="Per-task override such as MeetingBank=50 (repeatable).",
    )
    p.add_argument("--exact-budget", action="store_true")
    p.add_argument("--initial-points", type=int, default=16)
    p.add_argument("--bo-batch", type=int, default=4)
    p.add_argument("--ei-xi", type=float, default=0.05)
    p.add_argument("--ei-stop", type=float, default=0.05)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--candidate-pool", type=int, default=4096)
    p.add_argument("--probe-size", type=int, default=500)
    p.add_argument("--validation-replay-buffer", type=Path)
    p.add_argument(
        "--validation-replay-tasks", nargs="*",
        default=("NumGLUE-cm", "NumGLUE-ds"),
    )
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--validation-temperature", type=float, default=0.0)
    p.add_argument("--validation-samples", type=int, default=1)
    p.add_argument("--confirmation-fraction", type=float, default=0.0)
    p.add_argument("--confirmation-top-k", type=int, default=5)
    p.add_argument(
        "--confirmation-folds", type=int, default=1,
        help=(
            "Use deterministic K-fold candidate confirmation over the full "
            "validation pool. A value greater than one disables the fixed "
            "confirmation holdout."
        ),
    )
    p.add_argument(
        "--confirmation-full-validation", action="store_true",
        help=(
            "Search on the full validation pool, then rerank finalists using "
            "fresh repeated generations on that same full pool."
        ),
    )
    p.add_argument(
        "--include-task-vertex-finalist", action="store_true",
        help="Always add the evaluated task simplex vertex to BO finalists.",
    )
    p.add_argument("--confirmation-samples", type=int, default=8)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--bootstrap-confidence", type=float, default=0.90)
    p.add_argument(
        "--no-slow-fallback", action="store_true",
        help=(
            "Keep Slow-only as a diagnostic baseline, but never select it. "
            "Select the Fast finalist with the highest confirmation mean, "
            "regardless of its paired-bootstrap LCB."
        ),
    )
    p.add_argument("--heteroscedastic-noise", action="store_true")
    p.add_argument(
        "--include-data-noise", action="store_true",
        help=(
            "Add finite-validation-set uncertainty Var_x[E_seed score(x)]/n "
            "to the decoding-seed variance used as GP observation noise."
        ),
    )
    p.add_argument(
        "--paired-vertex-objective", action="store_true",
        help=(
            "Fit the GP to per-example score differences relative to the "
            "evaluated task vertex. Observation noise is estimated from "
            "paired decoding samples and a Bayesian bootstrap over examples."
        ),
    )
    p.add_argument("--adaptive-xi", action="store_true")
    p.add_argument(
        "--slow-test-results", type=Path,
        help=(
            "Optional JSON containing the existing SFT+Replay test_scores. "
            "When BO falls back to Slow-only, reuse those eight measurements "
            "instead of generating a statistically different duplicate."
        ),
    )
    p.add_argument(
        "--vertex-test-results-dir", type=Path,
        help=(
            "Optional directory containing TASK_vertex.json files from the "
            "final checkpoint evaluation. If BO selects the exact task "
            "vertex, reuse its eight test scores instead of regenerating an "
            "independent duplicate measurement."
        ),
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Reuse already completed task entries from --output.",
    )
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def actual_prompt_probe(rows, tokenizer, task, n, seed, max_tokens):
    """Filter with the prompt actually passed to generation, not gold PI."""
    indices = list(range(len(rows)))
    random.Random(f"{seed}:{task}").shuffle(indices)
    selected = []
    for index in indices:
        row = rows[index]
        content = ensure_task_prompt(task, str(row["prompt"]))
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        if len(tokenizer(
            rendered, add_special_tokens=False,
        ).input_ids) <= max_tokens:
            selected.append({**row, "source_index": index})
        if len(selected) == n:
            break
    return selected


def replay_rows_for_task(path, task):
    if path is None:
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    manifest = json.loads(path.with_name("buffer.manifest.json").read_text())
    allocation = manifest["allocation"]
    offset = 0
    result = []
    for source_task, count in allocation.items():
        block = rows[offset:offset + int(count)]
        if source_task == task:
            result.extend(block)
        offset += int(count)
    return result


def task_max_tokens(task):
    return 1 if task in {"C-STANCE", "FOMC"} else 512


def key(point):
    return tuple(np.round(np.asarray(point, dtype=np.float64), 8))


def load_slow_test_scores(path, task):
    if path is None:
        return None
    payload = json.loads(path.read_text())
    tasks = payload.get("tasks", payload)
    entry = tasks.get(task)
    if entry is None:
        return None
    scores = entry.get("test_scores", entry) if isinstance(entry, dict) else entry
    if not isinstance(scores, list) or len(scores) != 8:
        raise ValueError(
            f"slow-test result for {task} must contain exactly eight scores"
        )
    return [float(value) for value in scores]


def load_vertex_test_scores(directory, task):
    if directory is None:
        return None
    path = directory / f"{task}_vertex.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    scores = payload.get("scores")
    if not isinstance(scores, list) or len(scores) != 8:
        raise ValueError(
            f"vertex test result for {task} must contain exactly eight scores"
        )
    return [float(value) for value in scores]


def initial_design(dim, task_index, count, seed):
    points = [np.eye(dim)[task_index]]
    points.extend(np.eye(dim)[i] for i in range(dim) if i != task_index)
    points.append(np.full(dim, 1.0 / dim))
    needed = max(0, count - len(points))
    if needed:
        m = int(math.ceil(math.log2(needed)))
        u = qmc.Sobol(dim, scramble=True, seed=seed).random_base2(m)
        values = -np.log(np.clip(u[:needed], 1e-9, 1.0))
        points.extend(values / values.sum(1, keepdims=True))
    return points[:count]


def candidate_design(dim, count, seed, best):
    rng = np.random.default_rng(seed)
    dense = rng.dirichlet(np.ones(dim), count // 2)
    sparse = rng.dirichlet(np.full(dim, 0.3), count // 4)
    local_count = count - len(dense) - len(sparse)
    concentration = 24.0 * np.asarray(best) + 0.3
    local = rng.dirichlet(concentration, local_count)
    return np.concatenate((dense, sparse, local), axis=0)


def fit_gp(x, y, observation_variance=None):
    # sqrt(lambda) embeds the simplex with Hellinger geometry.
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
        length_scale=0.5, length_scale_bounds=(0.05, 5.0),
    )
    if observation_variance is None:
        kernel += WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 2.0))
        alpha = 1e-10
    else:
        # sklearn applies alpha after normalize_y, so express the variance in
        # normalized-score units. This is the uncertainty of the repeated-run
        # mean, not the variance of individual stochastic generations.
        score_variance = max(float(np.var(y)), 1e-6)
        alpha = np.maximum(np.asarray(observation_variance) / score_variance, 1e-6)
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=0,
        n_restarts_optimizer=1, alpha=alpha,
    )
    gp.fit(np.sqrt(np.clip(x, 0, 1)), y)
    return gp


def expected_improvement(gp, candidates, best, xi):
    mean, std = gp.predict(
        np.sqrt(np.clip(candidates, 0, 1)), return_std=True,
    )
    improvement = mean - best - xi
    z = improvement / np.maximum(std, 1e-9)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    ei[std < 1e-9] = 0
    return ei


def main():
    args = parse_args()
    overrides = {}
    for item in args.task_budget:
        name, value = item.rsplit("=", 1)
        if name not in TASKS:
            raise ValueError(f"unknown task budget override: {name}")
        overrides[name] = int(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.checkpoint / "simplex_config.json").read_text())
    dim = int(config["stage"]) + 1
    unseen = [task for task in args.tasks if TASKS.index(task) >= dim]
    if unseen:
        raise ValueError(f"checkpoint has {dim} tasks; unseen requested={unseen}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    embedding = load_embedding(args.checkpoint)
    state = torch.load(
        args.checkpoint / "simplex_soft_prompt.pt", map_location="cpu",
        weights_only=True,
    )
    prompt_model = SimplexBezierPrompt(
        dim, int(config["degree"]), int(config["prompt_length"]),
        int(config["hidden_size"]),
        residual_scale=float(config.get("residual_scale", 1.0)),
    )
    prompt_model.load_state_dict(state)
    prompt_model.eval()
    prompt_length = int(config["prompt_length"])
    eval_max_model_len = (
        EVAL_TEXT_PROMPT_LIMIT + prompt_length + EVAL_MAX_GENERATION
    )
    llm = LLM(
        model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=eval_max_model_len, max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False, enable_prompt_embeds=True,
        enforce_eager=True,
    )
    if args.resume and args.output.exists():
        result = json.loads(args.output.read_text())
    else:
        result = {
            "checkpoint": str(args.checkpoint),
            "search": "simplex GP Bayesian optimization with batched EI",
            "budget": args.budget, "task_budget_overrides": overrides,
            "validation_temperature": args.validation_temperature,
            "validation_samples": args.validation_samples,
            "confirmation_fraction": args.confirmation_fraction,
            "confirmation_top_k": args.confirmation_top_k,
            "confirmation_folds": args.confirmation_folds,
            "confirmation_full_validation": args.confirmation_full_validation,
            "include_task_vertex_finalist": args.include_task_vertex_finalist,
            "confirmation_samples": args.confirmation_samples,
            "heteroscedastic_noise": args.heteroscedastic_noise,
            "include_data_noise": args.include_data_noise,
            "paired_vertex_objective": args.paired_vertex_objective,
            "adaptive_xi": args.adaptive_xi,
            "tasks": {},
        }

    def save_result():
        """Persist each completed task so a later worker failure is resumable."""
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        os.replace(temporary, args.output)

    for task in args.tasks:
        if task in result["tasks"]:
            print(json.dumps({
                "event": "simplex_bo_resume_skip", "task": task,
            }), flush=True)
            continue
        task_budget = overrides.get(task, args.budget)
        task_index = TASKS.index(task)
        source = json.loads((args.data_root / task / "eval.json").read_text())
        validation = actual_prompt_probe(
            source, tokenizer, task, args.probe_size, args.seed, 2048,
        )
        if args.confirmation_folds < 1:
            raise ValueError("--confirmation-folds must be positive")
        if not 0.0 <= args.confirmation_fraction < 1.0:
            raise ValueError("--confirmation-fraction must be in [0, 1)")
        if args.confirmation_full_validation or args.confirmation_folds > 1:
            # BO uses the complete scarce validation split. Finalists are then
            # compared on deterministic folds rather than on a noisy 10-row
            # holdout (notably for NumGLUE-cm).
            search_validation = validation
            confirmation = validation
        else:
            confirmation_count = (
                max(1, int(round(len(validation) * args.confirmation_fraction)))
                if args.confirmation_fraction > 0 else 0
            )
            if confirmation_count:
                search_validation = validation[:-confirmation_count]
                confirmation = validation[-confirmation_count:]
            else:
                search_validation = validation
                confirmation = []
        replay = []
        if task in args.validation_replay_tasks:
            replay_source = replay_rows_for_task(args.validation_replay_buffer, task)
            replay = actual_prompt_probe(
                replay_source, tokenizer, task, 100000,
                args.seed + 17, 2048,
            )
        rows = [{**row, "task": task} for row in [*search_validation, *replay]]
        confirmation_pool = (
            [*confirmation, *replay]
            if args.confirmation_full_validation or args.confirmation_folds > 1
            else confirmation
        )
        confirmation_rows = [{**row, "task": task} for row in confirmation_pool]
        max_tokens = task_max_tokens(task)
        cache = {}
        per_example_cache = {}
        task_vertex_key = key(np.eye(dim, dtype=np.float64)[task_index])
        bootstrap_rng = np.random.default_rng(
            args.seed + 300007 + task_index,
        )
        bootstrap_weights = bootstrap_rng.dirichlet(
            np.ones(len(rows), dtype=np.float64),
            size=args.bootstrap_replicates,
        )

        plain_prompts = make_plain_inputs(
            tokenizer, rows, eval_max_model_len - max_tokens,
        )
        validation_sampling = SamplingParams(
            temperature=args.validation_temperature,
            max_tokens=max_tokens,
            n=args.validation_samples,
            seed=args.seed,
        )
        plain_outputs = llm.generate(
            plain_prompts, validation_sampling, use_tqdm=False,
        )
        slow_validation_scores = [repository_score(
            task,
            [item.outputs[i].text.strip() for item in plain_outputs],
            rows,
        ) for i in range(args.validation_samples)]
        slow_validation_score = sum(slow_validation_scores) / len(slow_validation_scores)
        slow_validation_std = math.sqrt(sum(
            (score - slow_validation_score) ** 2
            for score in slow_validation_scores
        ) / len(slow_validation_scores))
        print(json.dumps({
            "event": "simplex_bo_slow_only", "task": task,
            "score": slow_validation_score,
            "score_std": slow_validation_std,
            "scores": slow_validation_scores,
            "num_validation_rows": len(search_validation),
            "num_confirmation_rows": len(confirmation_rows),
            "num_replay_rows": len(replay),
        }), flush=True)

        def evaluate_batch(points):
            fresh = [np.asarray(p, dtype=np.float64) for p in points if key(p) not in cache]
            if not fresh:
                return
            fresh = fresh[:task_budget - len(cache)]
            all_prompts, spans = [], []
            for point in fresh:
                point = np.maximum(point, 0); point /= point.sum()
                with torch.no_grad():
                    memory = prompt_model(
                        torch.tensor(point, dtype=torch.float32).unsqueeze(0)
                    )[0].to(torch.bfloat16)
                prompts = make_inputs(
                    tokenizer, embedding, memory, rows,
                    eval_max_model_len - max_tokens - prompt_length, "chat_start",
                )
                spans.append((len(all_prompts), len(all_prompts) + len(prompts), point))
                all_prompts.extend(prompts)
            outputs = llm.generate(
                all_prompts, validation_sampling, use_tqdm=False,
            )
            for begin, end, point in spans:
                point_outputs = outputs[begin:end]
                scores = [repository_score(
                    task,
                    [item.outputs[i].text.strip() for item in point_outputs],
                    rows,
                ) for i in range(args.validation_samples)]
                per_example_scores = np.asarray([
                    repository_scores_per_example(
                        task,
                        [item.outputs[i].text.strip() for item in point_outputs],
                        rows,
                    )
                    for i in range(args.validation_samples)
                ], dtype=np.float64)
                score = sum(scores) / len(scores)
                score_std = math.sqrt(sum(
                    (sample_score - score) ** 2 for sample_score in scores
                ) / len(scores))
                # Two distinct uncertainties affect an aggregate validation
                # score.  Repeated generations estimate decoding noise, while
                # the variance across question-level expected scores estimates
                # uncertainty from observing a finite validation sample.  More
                # decoding samples cannot shrink the latter.
                # Preserve the previous population-variance estimator so that
                # --include-data-noise changes only the newly requested term.
                seed_variance_of_mean = score_std ** 2 / args.validation_samples
                per_example_means = per_example_scores.mean(axis=0)
                data_variance_of_mean = (
                    float(np.var(per_example_means, ddof=1)) / len(rows)
                    if len(rows) > 1 else 0.0
                )
                observation_variance = seed_variance_of_mean
                if args.include_data_noise:
                    observation_variance += data_variance_of_mean
                point_key = key(point)
                per_example_cache[point_key] = per_example_scores
                paired_delta_mean = None
                paired_seed_variance_of_mean = None
                paired_data_variance_of_mean = None
                if args.paired_vertex_objective:
                    vertex_scores = per_example_cache.get(task_vertex_key)
                    if vertex_scores is None:
                        raise RuntimeError(
                            "task vertex must be evaluated before other BO points"
                        )
                    paired_scores = per_example_scores - vertex_scores
                    paired_delta_mean = float(paired_scores.mean())
                    paired_seed_means = paired_scores.mean(axis=1)
                    paired_seed_std = float(np.std(paired_seed_means, ddof=0))
                    paired_seed_variance_of_mean = (
                        paired_seed_std ** 2 / args.validation_samples
                    )
                    paired_example_means = paired_scores.mean(axis=0)
                    bootstrap_means = bootstrap_weights @ paired_example_means
                    paired_data_variance_of_mean = float(
                        np.var(bootstrap_means, ddof=0)
                    )
                    observation_variance = (
                        paired_seed_variance_of_mean
                        + paired_data_variance_of_mean
                    )
                value = {
                    "preference": list(key(point)),
                    "score": score,
                    "score_std": score_std,
                    "seed_variance_of_mean": seed_variance_of_mean,
                    "data_variance_of_mean": data_variance_of_mean,
                    "paired_delta_mean": paired_delta_mean,
                    "paired_seed_variance_of_mean": paired_seed_variance_of_mean,
                    "paired_data_variance_of_mean": paired_data_variance_of_mean,
                    "score_variance_of_mean": max(
                        observation_variance, 0.05 ** 2,
                    ),
                    "scores": scores,
                    "mean_output_tokens": sum(
                        len(sample.token_ids)
                        for item in point_outputs for sample in item.outputs
                    ) / (len(rows) * args.validation_samples),
                }
                cache[key(point)] = value
                print(json.dumps({
                    "event": "simplex_bo_query", "task": task,
                    "query": len(cache), **value,
                }), flush=True)

        initial = initial_design(
            dim, task_index, min(args.initial_points, task_budget),
            args.seed + task_index,
        )
        for start in range(0, len(initial), args.bo_batch):
            evaluate_batch(initial[start:start + args.bo_batch])

        stale_rounds = 0
        round_id = 0
        while len(cache) < task_budget:
            round_id += 1
            values = list(cache.values())
            x = np.asarray([v["preference"] for v in values])
            objective_name = (
                "paired_delta_mean" if args.paired_vertex_objective else "score"
            )
            y = np.asarray([v[objective_name] for v in values])
            observation_variance = None
            if args.heteroscedastic_noise:
                observation_variance = np.asarray([
                    v["score_variance_of_mean"] for v in values
                ])
            gp = fit_gp(x, y, observation_variance)
            posterior_observed = gp.predict(np.sqrt(np.clip(x, 0, 1)))
            best_value = values[int(np.argmax(posterior_observed))]
            pool = candidate_design(
                dim, args.candidate_pool,
                args.seed + 1009 * task_index + 7919 * round_id,
                best_value["preference"],
            )
            pool = np.asarray([p for p in pool if key(p) not in cache])
            xi = args.ei_xi
            if args.adaptive_xi and observation_variance is not None:
                xi = max(xi, float(np.median(np.sqrt(observation_variance))))
            ei = expected_improvement(
                gp, pool, float(posterior_observed.max()), xi,
            )
            order = np.argsort(-ei)
            chosen = []
            # Diversity-aware batched EI under Hellinger distance.
            for index in order:
                candidate = pool[index]
                if all(np.linalg.norm(np.sqrt(candidate) - np.sqrt(p)) > 0.08 for p in chosen):
                    chosen.append(candidate)
                if len(chosen) >= min(args.bo_batch, task_budget - len(cache)):
                    break
            max_ei = float(ei[order[0]])
            stale_rounds = stale_rounds + 1 if max_ei < args.ei_stop else 0
            print(json.dumps({
                "event": "simplex_bo_round", "task": task,
                "round": round_id, "max_ei": max_ei,
                "stale_rounds": stale_rounds, "xi": xi,
            }), flush=True)
            if not args.exact_budget and stale_rounds >= args.early_stop_patience:
                break
            evaluate_batch(chosen)

        best_search = max(cache.values(), key=lambda value: value["score"])
        best = best_search
        selected_slow_only = slow_validation_score >= best_search["score"]
        confirmation_result = None
        if confirmation_rows:
            confirmation_sampling = SamplingParams(
                temperature=args.validation_temperature,
                max_tokens=max_tokens,
                n=args.confirmation_samples,
                seed=args.seed + 100003,
            )
            slow_confirmation_outputs = llm.generate(
                make_plain_inputs(
                    tokenizer, confirmation_rows, eval_max_model_len - max_tokens,
                ),
                confirmation_sampling,
                use_tqdm=False,
            )
            slow_predictions = [
                [item.outputs[i].text.strip() for item in slow_confirmation_outputs]
                for i in range(args.confirmation_samples)
            ]
            slow_confirmation_scores = [
                repository_score(task, predictions, confirmation_rows)
                for predictions in slow_predictions
            ]
            slow_confirmation_mean = float(np.mean(slow_confirmation_scores))

            finalists = sorted(
                cache.values(), key=lambda value: value["score"], reverse=True,
            )[:args.confirmation_top_k]
            if args.include_task_vertex_finalist:
                task_vertex_key = key(np.eye(dim, dtype=np.float64)[task_index])
                task_vertex = cache.get(task_vertex_key)
                if task_vertex is None:
                    raise RuntimeError(
                        "task vertex is missing from the initial BO design"
                    )
                if all(
                    key(item["preference"]) != task_vertex_key
                    for item in finalists
                ):
                    finalists.append(task_vertex)
            all_confirmation_prompts, finalist_spans = [], []
            for finalist in finalists:
                point = torch.tensor(finalist["preference"], dtype=torch.float32)
                with torch.no_grad():
                    memory = prompt_model(point.unsqueeze(0))[0].to(torch.bfloat16)
                prompts = make_inputs(
                    tokenizer, embedding, memory, confirmation_rows,
                    eval_max_model_len - max_tokens - prompt_length, "chat_start",
                )
                finalist_spans.append((
                    len(all_confirmation_prompts),
                    len(all_confirmation_prompts) + len(prompts),
                    finalist,
                ))
                all_confirmation_prompts.extend(prompts)
            all_confirmation_outputs = llm.generate(
                all_confirmation_prompts,
                confirmation_sampling,
                use_tqdm=False,
            )

            rng = np.random.default_rng(args.seed + 200003 + task_index)
            bootstrap_indices = rng.integers(
                0, len(confirmation_rows),
                size=(args.bootstrap_replicates, len(confirmation_rows)),
            )
            slow_per_example = np.mean(np.asarray([
                repository_scores_per_example(
                    task, predictions, confirmation_rows,
                )
                for predictions in slow_predictions
            ]), axis=0)
            slow_bootstrap_scores = slow_per_example[bootstrap_indices].mean(axis=1)

            fold_indices = None
            slow_fold_scores = None
            if args.confirmation_folds > 1:
                if len(confirmation_rows) < args.confirmation_folds:
                    raise ValueError(
                        "confirmation pool is smaller than --confirmation-folds"
                    )
                fold_rng = np.random.default_rng(
                    args.seed + 300003 + task_index,
                )
                permutation = fold_rng.permutation(len(confirmation_rows))
                fold_indices = [
                    indices.tolist()
                    for indices in np.array_split(
                        permutation, args.confirmation_folds,
                    )
                ]
                slow_fold_scores = []
                for indices in fold_indices:
                    fold_rows = [confirmation_rows[i] for i in indices]
                    per_sample = [
                        repository_score(
                            task,
                            [predictions[i] for i in indices],
                            fold_rows,
                        )
                        for predictions in slow_predictions
                    ]
                    slow_fold_scores.append(float(np.mean(per_sample)))
                slow_confirmation_mean = float(np.mean(slow_fold_scores))

            confirmed = []
            for begin, end, finalist in finalist_spans:
                candidate_outputs = all_confirmation_outputs[begin:end]
                candidate_predictions = [
                    [item.outputs[i].text.strip() for item in candidate_outputs]
                    for i in range(args.confirmation_samples)
                ]
                candidate_scores = [
                    repository_score(task, predictions, confirmation_rows)
                    for predictions in candidate_predictions
                ]
                candidate_mean = float(np.mean(candidate_scores))
                candidate_fold_scores = None
                if fold_indices is not None:
                    candidate_fold_scores = []
                    for indices in fold_indices:
                        fold_rows = [confirmation_rows[i] for i in indices]
                        per_sample = [
                            repository_score(
                                task,
                                [predictions[i] for i in indices],
                                fold_rows,
                            )
                            for predictions in candidate_predictions
                        ]
                        candidate_fold_scores.append(float(np.mean(per_sample)))
                    candidate_mean = float(np.mean(candidate_fold_scores))
                candidate_per_example = np.mean(np.asarray([
                    repository_scores_per_example(
                        task, predictions, confirmation_rows,
                    )
                    for predictions in candidate_predictions
                ]), axis=0)
                bootstrap_differences = (
                    candidate_per_example[bootstrap_indices].mean(axis=1)
                    - slow_bootstrap_scores
                )
                lcb_quantile = 1.0 - args.bootstrap_confidence
                difference_lcb = float(np.quantile(
                    bootstrap_differences, lcb_quantile,
                ))
                item = {
                    "preference": finalist["preference"],
                    "search_score": finalist["score"],
                    "confirmation_score": candidate_mean,
                    "confirmation_scores": candidate_scores,
                    "fold_scores": candidate_fold_scores,
                    "gain_over_slow": candidate_mean - slow_confirmation_mean,
                    "gain_lcb": difference_lcb,
                    "passes_lcb": difference_lcb > 0.0,
                }
                confirmed.append(item)
                print(json.dumps({
                    "event": "simplex_bo_confirmation", "task": task, **item,
                }), flush=True)

            if args.no_slow_fallback:
                selected_confirmation = max(
                    confirmed, key=lambda item: item["confirmation_score"],
                )
                best = cache[key(selected_confirmation["preference"])]
                selected_slow_only = False
            else:
                eligible = [item for item in confirmed if item["passes_lcb"]]
                if eligible:
                    selected_confirmation = max(
                        eligible, key=lambda item: item["confirmation_score"],
                    )
                    best = cache[key(selected_confirmation["preference"])]
                    selected_slow_only = False
                else:
                    selected_confirmation = None
                    selected_slow_only = True
            confirmation_result = {
                "num_rows": len(confirmation_rows),
                "num_samples": args.confirmation_samples,
                "num_folds": args.confirmation_folds,
                "slow_score": slow_confirmation_mean,
                "slow_scores": slow_confirmation_scores,
                "slow_fold_scores": slow_fold_scores,
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_confidence": args.bootstrap_confidence,
                "finalists": confirmed,
                "selected": selected_confirmation,
            }
            print(json.dumps({
                "event": "simplex_bo_confirmation_selected", "task": task,
                "selected_slow_only": selected_slow_only,
                "slow_score": slow_confirmation_mean,
                "selected": selected_confirmation,
            }), flush=True)
        test_source = json.loads((args.data_root / task / "test.json").read_text())
        test_rows = [{**row, "task": task} for row in actual_prompt_probe(
            test_source, tokenizer, task, 100000, args.seed, 2048,
        )]
        reused_slow_scores = (
            load_slow_test_scores(args.slow_test_results, task)
            if selected_slow_only else None
        )
        task_vertex = np.eye(len(TASKS), dtype=np.float64)[TASKS.index(task)]
        selected_exact_vertex = (
            not selected_slow_only
            and np.allclose(
                np.asarray(best["preference"], dtype=np.float64),
                task_vertex, rtol=0.0, atol=1e-8,
            )
        )
        reused_vertex_scores = (
            load_vertex_test_scores(args.vertex_test_results_dir, task)
            if selected_exact_vertex else None
        )
        reused_scores = reused_slow_scores or reused_vertex_scores
        if reused_scores is None and selected_slow_only:
            prompts = make_plain_inputs(
                tokenizer, test_rows, eval_max_model_len - max_tokens,
            )
        elif reused_scores is None:
            selected = torch.tensor(best["preference"], dtype=torch.float32)
            with torch.no_grad():
                memory = prompt_model(selected.unsqueeze(0))[0].to(torch.bfloat16)
            prompts = make_inputs(
                tokenizer, embedding, memory, test_rows,
                eval_max_model_len - max_tokens - prompt_length, "chat_start",
            )
        if reused_scores is None:
            outputs = llm.generate(
                prompts, SamplingParams(
                    temperature=0.1, max_tokens=max_tokens, n=8, seed=args.seed,
                ), use_tqdm=False,
            )
            scores = [repository_score(
                task, [item.outputs[i].text.strip() for item in outputs], test_rows,
            ) for i in range(8)]
        else:
            scores = reused_scores
            print(json.dumps({
                "event": (
                    "simplex_bo_reuse_slow_test" if selected_slow_only
                    else "simplex_bo_reuse_vertex_test"
                ),
                "task": task,
                "source": str(
                    args.slow_test_results if selected_slow_only
                    else args.vertex_test_results_dir / f"{task}_vertex.json"
                ),
                "test_scores": scores,
            }), flush=True)
        mean = sum(scores) / len(scores)
        std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
        result["tasks"][task] = {
            "num_validation_rows": len(search_validation),
            "num_confirmation_rows": len(confirmation_rows),
            "num_validation_replay_rows": len(replay),
            "slow_only_validation_score": slow_validation_score,
            "slow_only_validation_score_std": slow_validation_std,
            "slow_only_validation_scores": slow_validation_scores,
            "selected_slow_only": selected_slow_only,
            "selected_preference": None if selected_slow_only else best["preference"],
            "best_fast_preference": best_search["preference"],
            "best_fast_validation_score": best_search["score"],
            "selected_validation_score": (
                slow_validation_score if selected_slow_only else best["score"]
            ),
            "confirmation": confirmation_result,
            "num_queries": len(cache), "evaluations": list(cache.values()),
            "test_score": mean, "test_score_std": std,
            "test_scores": scores, "num_test_rows": len(test_rows),
        }
        save_result()
        print(json.dumps({
            "event": "simplex_bo_selected", "task": task,
            "selected_slow_only": selected_slow_only,
            "selected_preference": None if selected_slow_only else best["preference"],
            "best_fast_preference": best_search["preference"],
            "slow_only_validation_score": slow_validation_score,
            "validation_score": (
                slow_validation_score if selected_slow_only else best["score"]
            ),
            "test_score": mean, "test_score_std": std,
            "num_queries": len(cache),
        }), flush=True)

    save_result()
    engine_core = getattr(llm.llm_engine, "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        engine_core.shutdown()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
