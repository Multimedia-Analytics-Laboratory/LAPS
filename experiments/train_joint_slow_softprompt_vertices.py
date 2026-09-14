#!/usr/bin/env python3
"""Separated endpoint/slow learning for the TRACE task stream.

At every stage, completion-only SFT first freezes the slow backbone and fits
the new task vertex for the task's full epoch schedule.  The fitted vertex is
then frozen while the same SFT schedule updates the slow backbone.
Predictor--corrector blocks interleaved with the slow phase freeze the backbone
and transport historical vertices with Teacher||Student full-vocabulary KL on
current-student rollouts. Historical KL deliberately updates only historical
vertices, never the slow backbone, current vertex, or non-vertex controls.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src.preference_memory import insert_latents, selective_log_softmax  # noqa: E402
from src.simplex_bezier_prompt import SimplexBezierPrompt  # noqa: E402
from train_trace_simplex_stch_stage import (  # noqa: E402
    batch_tensors,
    completion_logits,
)
from train_trace_opr_stch_mgda_stage import (  # noqa: E402
    EPOCHS,
    MAX_COMPLETION,
    TASKS,
    bucketed_epoch_order,
    make_scheduler,
    plain_completion_logits,
    prepare_baseline_sft_rows,
    prepare_buffer_rows,
    prepare_current_rows,
)


# TRACE's official generation protocol uses one token for the two
# classification tasks and up to 512 tokens for every later generation task.
# This is deliberately separate from MAX_COMPLETION, which controls how gold
# SFT targets are truncated.
REPLAY_ROLLOUT_TOKENS = {
    task: (1 if task in {"C-STANCE", "FOMC"} else 512)
    for task in TASKS
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, help="Zero-based TRACE stage index")
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--model", required=True)
    p.add_argument("--teacher-model")
    p.add_argument("--previous-prompt", type=Path)
    p.add_argument("--teacher-prompt", type=Path)
    p.add_argument(
        "--teacher-slow-only-tasks", nargs="*", default=[],
        help=(
            "Historical task names whose frozen teacher omits its task "
            "vertex. The current student still uses and updates that task's "
            "historical vertex."
        ),
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--buffer", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--global-batch", type=int, default=128)
    p.add_argument("--replay-global-batch", type=int, default=64)
    p.add_argument("--max-train-samples", type=int, default=5000)
    p.add_argument("--prompt-length", type=int, default=32)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--slow-lr", type=float, default=1e-5)
    p.add_argument("--new-prompt-lr", type=float, default=5e-4)
    p.add_argument("--old-prompt-lr", type=float, default=1e-3)
    p.add_argument(
        "--historical-vertex-lr", action="append", default=[], metavar="TASK=LR",
        help=(
            "Override the rebase learning rate for one historical task vertex. "
            "May be repeated; unspecified vertices use --old-prompt-lr."
        ),
    )
    p.add_argument(
        "--current-vertex-lr-scheduler-type",
        choices=("constant", "linear"), default="constant",
        help="Schedule applied only to the newly learned task vertex.",
    )
    p.add_argument("--slow-block-steps", type=int, default=8)
    p.add_argument("--historical-fkl-steps", type=int, default=4)
    p.add_argument("--final-historical-fkl-steps", type=int, default=10)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--condition-chunk", type=int, default=8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--transport-only-steps", type=int, default=0)
    p.add_argument("--post-slow-current-vertex-steps", type=int, default=0)
    p.add_argument("--post-slow-current-vertex-lr", type=float, default=0.03)
    p.add_argument(
        "--slow-replay-sft", action="store_true",
        help=(
            "Also train the plain Slow backbone on gold replay. Replay never "
            "updates the current or historical prompt vertices. Independently "
            "sampled replay token sums are importance-corrected to match the "
            "concatenated SFT+Replay dataset ratio."
        ),
    )
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--init-text",
        default="Answer the question accurately and follow the required output format.",
    )
    return p.parse_args()


def init_embedding(model, tokenizer, text, length):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    ids = (ids * math.ceil(length / len(ids)))[:length]
    values = torch.tensor(ids, dtype=torch.long, device=model.device)
    return model.get_input_embeddings()(values).detach().float().cpu()


def allreduce_grads(parameters):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    world = dist.get_world_size()
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


def clip_prompt_vertex_groups_(prompt_model, stage, max_norm):
    """Independently clip current-task and historical-vertex gradients.

    The two objectives have disjoint Bernstein support at simplex vertices:
    current-task SFT updates only the newly introduced vertex, while
    historical FKL updates only old vertices. Clipping the dense controls
    tensor as one parameter would nevertheless couple their effective step
    sizes through the shared global norm. Keep the two optimization blocks
    independent and explicitly discard numerical gradients on non-vertices.
    """
    grad = prompt_model.controls.grad
    zero = torch.zeros((), device=prompt_model.controls.device, dtype=torch.float32)
    if grad is None:
        return {
            "current_raw_norm": zero,
            "historical_raw_norm": zero,
            "current_clip_scale": zero,
            "historical_clip_scale": zero,
        }

    vertex_mask = (prompt_model.multi_indices == prompt_model.degree).sum(dim=1).eq(1)
    vertex_task = prompt_model.multi_indices.argmax(dim=1)
    current_mask = vertex_mask & vertex_task.eq(stage)
    historical_mask = vertex_mask & vertex_task.lt(stage)
    active_mask = current_mask | historical_mask
    grad[~active_mask] = 0

    def clip_slice(mask):
        if not bool(mask.any()):
            return zero, torch.ones_like(zero)
        raw_norm = torch.linalg.vector_norm(grad[mask].float())
        scale = (float(max_norm) / (raw_norm + 1e-6)).clamp(max=1.0)
        # Boolean advanced indexing does not return a writable view, so assign
        # the scaled values back explicitly.
        grad[mask] = grad[mask] * scale.to(grad.dtype)
        return raw_norm, scale

    current_norm, current_scale = clip_slice(current_mask)
    historical_norm, historical_scale = clip_slice(historical_mask)
    return {
        "current_raw_norm": current_norm,
        "historical_raw_norm": historical_norm,
        "current_clip_scale": current_scale,
        "historical_clip_scale": historical_scale,
    }


def epoch_batches(rows, epoch, seed, global_batch, rank, world,
                  steps_per_epoch=None):
    order = bucketed_epoch_order(rows, seed, epoch, 128)
    steps = (
        math.ceil(len(order) / global_batch)
        if steps_per_epoch is None else int(steps_per_epoch)
    )
    padded = [order[index % len(order)] for index in range(steps * global_batch)]
    local = global_batch // world
    for step in range(steps):
        block = padded[step * global_batch:(step + 1) * global_batch]
        yield [rows[index] for index in block[rank * local:(rank + 1) * local]]


def cyclic_rows(rows, count, step, rank, world, seed):
    rng = random.Random(seed + 104729 * (step // max(1, math.ceil(len(rows) / (count * world)))))
    order = list(range(len(rows)))
    rng.shuffle(order)
    offset = step * count * world + rank * count
    return [rows[order[(offset + index) % len(order)]] for index in range(count)]


def cyclic_global_rows(rows, global_count, step, rank, world, seed, rank_shift=0):
    """Deterministically partition a possibly sub-world-size global batch."""
    logical_rank = (rank - rank_shift) % world
    base, remainder = divmod(global_count, world)
    local_count = base + int(logical_rank < remainder)
    prefix = logical_rank * base + min(logical_rank, remainder)
    if local_count == 0:
        return []
    cycle_steps = max(1, math.ceil(len(rows) / global_count))
    rng = random.Random(seed + 104729 * (step // cycle_steps))
    order = list(range(len(rows)))
    rng.shuffle(order)
    offset = step * global_count + prefix
    return [rows[order[(offset + index) % len(order)]] for index in range(local_count)]


def endpoint(prompt_model, task, count, device):
    pref = torch.zeros(count, prompt_model.num_tasks, device=device)
    if torch.is_tensor(task):
        task = task.to(device=device, dtype=torch.long).reshape(-1)
        if task.numel() != count:
            raise ValueError(f"Expected {count} task ids, received {task.numel()}")
        pref.scatter_(1, task[:, None], 1.0)
    else:
        pref[:, int(task)] = 1.0
    return prompt_model(pref)


def prompt_generate(model, prompt_model, tensors, task, tokenizer, max_new_tokens):
    code = endpoint(prompt_model, task, len(tensors["task_ids"]), model.device).detach()
    embeds, mask = insert_latents(
        model.get_input_embeddings(), tensors["prompt_ids"],
        tensors["prompt_mask"], code, tensors["positions"],
    )
    old_cache = model.config.use_cache
    model.config.use_cache = True
    was_training = model.training
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            inputs_embeds=embeds, attention_mask=mask, do_sample=True,
            temperature=1.0, top_p=1.0, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, use_cache=True,
        )
    model.config.use_cache = old_cache
    model.train(was_training)
    completion = generated[:, -max_new_tokens:].contiguous()
    # Include the first EOS token in the KL, then mask padding/anything after
    # it.  This remains correct when pad_token_id == eos_token_id.
    positions = torch.arange(completion.shape[1], device=completion.device)[None]
    is_eos = completion.eq(tokenizer.eos_token_id)
    sentinel = torch.full_like(positions.expand_as(completion), completion.shape[1])
    first_eos = torch.where(is_eos, positions, sentinel).amin(dim=1)
    completion_mask = positions.le(first_eos[:, None]).to(completion.dtype)
    return completion, completion_mask


def replay_task_groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault(str(row["task"]), []).append(row)
    return [(task, groups[task]) for task in TASKS if task in groups]


@torch.no_grad()
def teacher_logits(
    teacher, teacher_prompt, tensors, completion, mask, *, use_vertex=True,
):
    if not use_vertex:
        return plain_completion_logits(
            teacher, tensors["prompt_ids"], tensors["prompt_mask"],
            completion, mask,
        )
    code = endpoint(
        teacher_prompt, tensors["task_ids"],
        len(tensors["task_ids"]), teacher.device,
    )
    return completion_logits(
        teacher, tensors["prompt_ids"], tensors["prompt_mask"],
        completion, mask, code, tensors["positions"],
    ).float()


def full_fkl(teacher_values, student_values, mask):
    teacher_logp = F.log_softmax(teacher_values.float(), dim=-1)
    student_logp = F.log_softmax(student_values.float(), dim=-1)
    token = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)
    return (token * mask).sum() / mask.sum().clamp_min(1)


def sft_backward(model, prompt_model, tensors, task, chunk, denominator, scale=1.0):
    """Token-sum SFT backward normalized by a shared global-token denominator."""
    total = len(tensors["task_ids"])
    nll_sum = torch.zeros((), device=model.device)
    token_count = torch.zeros((), device=model.device)
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        code = endpoint(prompt_model, task, end - start, model.device)
        logits = completion_logits(
            model, tensors["prompt_ids"][start:end], tensors["prompt_mask"][start:end],
            tensors["target_ids"][start:end], tensors["target_mask"][start:end],
            code, tensors["positions"][start:end],
        )
        mask = tensors["target_mask"][start:end]
        token_nll = -selective_log_softmax(
            logits, tensors["target_ids"][start:end],
        ).float() * mask
        chunk_sum = token_nll.sum()
        (chunk_sum * scale / denominator).backward()
        nll_sum += chunk_sum.detach()
        token_count += mask.sum().detach()
        del logits, token_nll
    return nll_sum, token_count


def plain_replay_sft_backward(
    model, tensors, chunk, denominator, importance_weight,
):
    """Backpropagate replay SFT into Slow only, with no Fast prompt graph."""
    total = len(tensors["task_ids"])
    nll_sum = torch.zeros((), device=model.device)
    token_count = torch.zeros((), device=model.device)
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        logits = plain_completion_logits(
            model,
            tensors["prompt_ids"][start:end],
            tensors["prompt_mask"][start:end],
            tensors["target_ids"][start:end],
            tensors["target_mask"][start:end],
        )
        mask = tensors["target_mask"][start:end]
        token_nll = -selective_log_softmax(
            logits, tensors["target_ids"][start:end],
        ).float() * mask
        chunk_sum = token_nll.sum()
        (chunk_sum * importance_weight / denominator).backward()
        nll_sum += chunk_sum.detach()
        token_count += mask.sum().detach()
        del logits, token_nll
    return nll_sum, token_count


def load_rows(args, tokenizer, task, stage):
    prep = SimpleNamespace(
        task=task, stage=stage, data_root=args.data_root, buffer=args.buffer,
        max_train_samples=args.max_train_samples, max_prompt_length=args.max_prompt_length,
        max_completion_length=MAX_COMPLETION[task], acquire_objective="sft",
        replay_overlength="head_tail",
        seed=args.seed,
    )
    prepared_current, dropped = prepare_current_rows(prep, tokenizer)
    replay = prepare_buffer_rows(prep, tokenizer)
    current, current_sft_dropped = prepare_baseline_sft_rows(
        prepared_current, tokenizer, max_length=args.max_prompt_length,
    )
    replay_sft, replay_sft_dropped = prepare_baseline_sft_rows(
        replay, tokenizer, max_length=args.max_prompt_length,
    )
    return (
        current, replay, replay_sft, dropped + current_sft_dropped,
        current_sft_dropped, replay_sft_dropped,
    )


def save_stage(args, model, prompt_model, tokenizer, accelerator, metrics):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(prompt_model.state_dict(), args.output_dir / "simplex_soft_prompt.pt")
        (args.output_dir / "simplex_config.json").write_text(json.dumps({
            "stage": int(args.stage_index),
            "degree": args.degree,
            "prompt_length": args.prompt_length,
            "hidden_size": int(model.config.hidden_size),
            "residual_scale": 1.0,
            "soft_prompt_placement": "chat_start",
        }, indent=2) + "\n")
        (args.output_dir / "train_config.json").write_text(
            json.dumps({**vars(args), **metrics}, indent=2, default=str) + "\n"
        )
        (args.output_dir / "STAGE_COMPLETE").touch()
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")
    rank, world, device = accelerator.process_index, accelerator.num_processes, accelerator.device
    if args.global_batch % world or args.replay_global_batch % world:
        raise ValueError("global batches must be divisible by world size")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    legacy = {"cstance": 0, "fomc": 1}
    stage = legacy.get(str(args.stage), int(args.stage) if str(args.stage).isdigit() else -1)
    if not 0 <= stage < len(TASKS):
        raise ValueError(f"Invalid TRACE stage: {args.stage}")
    task = args.task or TASKS[stage]
    if task != TASKS[stage]:
        raise ValueError(f"stage {stage} must train {TASKS[stage]}, not {task}")
    args.stage_index = stage
    historical_vertex_lr_overrides = {}
    for specification in args.historical_vertex_lr:
        try:
            task_name, value = specification.rsplit("=", 1)
            learning_rate = float(value)
        except ValueError as error:
            raise ValueError(
                "--historical-vertex-lr must have the form TASK=LR"
            ) from error
        if task_name not in TASKS:
            raise ValueError(f"unknown historical task in LR override: {task_name}")
        if learning_rate < 0:
            raise ValueError("historical vertex learning rates must be nonnegative")
        historical_vertex_lr_overrides[task_name] = learning_rate
    epochs = EPOCHS[task]
    (
        current, replay, replay_sft, dropped,
        current_sft_dropped, replay_sft_dropped,
    ) = load_rows(args, tokenizer, task, stage)
    post_slow_mode = (
        args.transport_only_steps > 0
        or args.post_slow_current_vertex_steps > 0
    )
    needs_historical_teacher = stage > 0 and (
        not post_slow_mode or args.transport_only_steps > 0
    )
    if post_slow_mode:
        if not args.previous_prompt:
            raise ValueError("post-Slow mode requires the same-stage --previous-prompt")
        state = torch.load(args.previous_prompt, map_location="cpu", weights_only=True)
        prompt_model = SimplexBezierPrompt(
            stage + 1, args.degree, args.prompt_length, int(model.config.hidden_size),
        ).to(device)
        prompt_model.load_state_dict(state)
    elif stage == 0:
        prompt_model = SimplexBezierPrompt(
            1, args.degree, args.prompt_length, int(model.config.hidden_size),
            random_init=True,
        ).to(device)
    else:
        if not args.previous_prompt or not args.teacher_model or not args.buffer:
            raise ValueError("Stages after C-STANCE require a teacher, previous prompt, and replay buffer")
        old_state = torch.load(args.previous_prompt, map_location="cpu", weights_only=True)
        prompt_model = SimplexBezierPrompt.expand_from_state(
            old_state, stage + 1, random_init=True,
        ).to(device)
    slow_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.slow_lr, weight_decay=0.0, fused=True,
    )
    prompt_optimizer = torch.optim.AdamW(
        prompt_model.parameters(), lr=args.new_prompt_lr, weight_decay=0.0, fused=True,
    )
    effective_sft_rows = len(current) + (
        len(replay_sft) if args.slow_replay_sft else 0
    )
    steps_per_epoch = math.ceil(effective_sft_rows / args.global_batch)
    total_steps = steps_per_epoch * epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    slow_scheduler = make_scheduler(slow_optimizer, total_steps, 0, "linear")
    teacher = teacher_prompt = None
    teacher_slow_only_tasks = set(args.teacher_slow_only_tasks)
    if post_slow_mode and teacher_slow_only_tasks:
        raise ValueError(
            "post-Slow transport fixes teacher semantics to the previous "
            "Slow plus its historical vertex; slow-only teacher switching is disabled"
        )
    unknown_teacher_tasks = teacher_slow_only_tasks.difference(TASKS[:stage])
    if unknown_teacher_tasks:
        raise ValueError(
            "--teacher-slow-only-tasks contains non-historical tasks: "
            + ", ".join(sorted(unknown_teacher_tasks))
        )
    if needs_historical_teacher:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to(device).eval().requires_grad_(False)
        teacher.config.use_cache = False
        teacher_prompt_path = args.teacher_prompt or args.previous_prompt
        old_state = torch.load(teacher_prompt_path, map_location="cpu", weights_only=True)
        teacher_prompt = SimplexBezierPrompt(
            stage, args.degree, args.prompt_length, int(model.config.hidden_size),
        ).to(device)
        teacher_prompt.load_state_dict(old_state)
        teacher_prompt.requires_grad_(False).eval()

    if rank == 0:
        print(json.dumps({
            "event": "start", "stage": stage, "task": task, "rows": len(current),
            "replay_rows": len(replay), "replay_sft_rows": len(replay_sft),
            "current_sft_total_length_dropped": current_sft_dropped,
            "replay_sft_dropped": replay_sft_dropped,
            "dropped": dropped, "epochs": epochs,
            "total_steps": total_steps, "world": world, "global_batch": args.global_batch,
            "effective_sft_rows": effective_sft_rows,
            "steps_per_epoch": steps_per_epoch,
            "slow_lr": args.slow_lr, "new_prompt_lr": args.new_prompt_lr,
            "old_prompt_lr": args.old_prompt_lr, "placement": "chat_start",
            "historical_vertex_lr_overrides": historical_vertex_lr_overrides,
            "prompt_initialization": "independent_normal_0_1",
            "slow_block_steps": args.slow_block_steps,
            "historical_fkl_steps": args.historical_fkl_steps,
            "final_historical_fkl_steps": args.final_historical_fkl_steps,
            "current_vertex_schedule": "vertex_then_slow",
            "current_vertex_steps": total_steps,
            "slow_steps": total_steps,
            "historical_fkl_reduction": "equal_vertex_shared_anchor_token_mean",
            "slow_replay_sft": args.slow_replay_sft,
            "slow_replay_weight": (
                len(replay_sft) / len(current)
                if args.slow_replay_sft and replay_sft else 0.0
            ),
            "historical_objective": (
                "task_length_student_rollout_full_vocab_teacher_to_student_fkl"
                if needs_historical_teacher else None
            ),
            "teacher_slow_only_tasks": sorted(teacher_slow_only_tasks),
            "post_slow_current_vertex_steps": args.post_slow_current_vertex_steps,
            "post_slow_current_vertex_lr": args.post_slow_current_vertex_lr,
            "historical_anchor_protocol": (
                "global_8_shared_4_replay_plus_4_current"
                if needs_historical_teacher else None
            ),
        }), flush=True)
    if min(
        args.slow_block_steps, args.historical_fkl_steps,
        args.final_historical_fkl_steps,
    ) < 0 or args.slow_block_steps == 0:
        raise ValueError("transport block sizes must be non-negative and slow-block-steps > 0")

    started = time.monotonic()
    vertex_step = 0
    global_step = 0
    transport_step = 0
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    vertex_mask = (prompt_model.multi_indices == args.degree).sum(dim=1).eq(1)
    vertex_task = prompt_model.multi_indices.argmax(dim=1)
    current_vertex_mask = vertex_mask & vertex_task.eq(stage)
    historical_vertex_mask = vertex_mask & vertex_task.lt(stage)

    def prompt_step_with_mask(active_mask, learning_rate, row_learning_rates=None):
        """Apply AdamW only to selected control rows, including momentum."""
        inactive_mask = ~active_mask
        before_values = prompt_model.controls.detach().clone()
        optimizer_lr = learning_rate
        if row_learning_rates is not None and bool(active_mask.any()):
            optimizer_lr = float(row_learning_rates[active_mask].max())
        for group in prompt_optimizer.param_groups:
            group["lr"] = optimizer_lr
        prompt_optimizer.step()
        # Adam momentum can move a row even when its current gradient is zero.
        # Restore inactive rows so the two phases are functionally disjoint.
        with torch.no_grad():
            prompt_model.controls[inactive_mask].copy_(before_values[inactive_mask])
            if row_learning_rates is not None and optimizer_lr > 0:
                ratios = (
                    row_learning_rates[active_mask] / optimizer_lr
                ).to(prompt_model.controls.dtype)
                shape = (len(ratios),) + (1,) * (prompt_model.controls.ndim - 1)
                delta = (
                    prompt_model.controls[active_mask]
                    - before_values[active_mask]
                )
                prompt_model.controls[active_mask].copy_(
                    before_values[active_mask] + delta * ratios.reshape(shape)
                )

    def current_vertex_learning_rate(step):
        if args.current_vertex_lr_scheduler_type == "constant":
            return float(args.new_prompt_lr)
        if total_steps <= 1:
            return 0.0
        # The first update uses the configured peak and the final update is
        # exactly zero, with no warmup for the current-task vertex.
        return float(args.new_prompt_lr) * (
            float(total_steps - step) / float(total_steps - 1)
        )

    def historical_fkl_update(phase):
        nonlocal transport_step
        if stage == 0:
            return
        transport_step += 1
        # Slow SFT keeps all prompt controls frozen. Temporarily expose the
        # prompt graph only for historical-vertex transport.
        prompt_model.requires_grad_(True)
        prompt_optimizer.zero_grad(set_to_none=True)
        if args.replay_global_batch != 8:
            raise ValueError(
                "shared-anchor transport currently requires replay-global-batch=8"
            )
        replay_rows = cyclic_global_rows(
            replay, 4, transport_step - 1,
            rank, world, args.seed + 991,
        )
        current_anchor_rows = cyclic_global_rows(
            current, 4, transport_step - 1,
            rank, world, args.seed + 1991, rank_shift=world // 2,
        )
        shared_rows = replay_rows + current_anchor_rows

        # The same eight global inputs support every historical endpoint.
        # Each endpoint generates its own on-policy trajectory, and its exact
        # frozen predecessor (old Slow + old endpoint) supplies the teacher.
        generated_groups = []
        for input_task, task_rows in replay_task_groups(shared_rows):
            rt = batch_tensors(task_rows, tokenizer, device)
            rows_in_group = len(rt["task_ids"])
            expanded = {
                key: (
                    value.repeat((stage,) + (1,) * (value.ndim - 1))
                    if torch.is_tensor(value) else value
                )
                for key, value in rt.items()
            }
            forced_ids = torch.arange(
                stage, device=device, dtype=torch.long,
            ).repeat_interleave(rows_in_group)
            completion, completion_mask = prompt_generate(
                model, prompt_model, expanded, forced_ids, tokenizer,
                REPLAY_ROLLOUT_TOKENS[input_task],
            )
            generated_groups.append((
                input_task, expanded, forced_ids, rows_in_group,
                completion, completion_mask,
            ))

        token_count_by_vertex = torch.zeros(
            stage, device=device, dtype=torch.float32,
        )
        for _, _, _, rows_in_group, _, mask in generated_groups:
            token_count_by_vertex += mask.reshape(
                stage, rows_in_group, -1,
            ).sum(dim=(1, 2)).float()
        if dist.is_initialized():
            dist.all_reduce(token_count_by_vertex, op=dist.ReduceOp.SUM)
        token_normalizer_by_vertex = token_count_by_vertex.clamp_min(1) / world

        model.requires_grad_(False)
        replay_value_sum = torch.zeros((), device=device)
        task_fkl_numerators = torch.zeros(stage, device=device, dtype=torch.float64)
        task_token_counts = torch.zeros(stage, device=device, dtype=torch.float64)
        task_sample_counts = torch.zeros(stage, device=device, dtype=torch.float64)
        for _, rt, forced_ids, rows_in_group, completion, completion_mask in generated_groups:
          for historical_task_id in range(stage):
            vertex_start = historical_task_id * rows_in_group
            vertex_end = vertex_start + rows_in_group
            for start in range(vertex_start, vertex_end, args.condition_chunk):
                end = min(start + args.condition_chunk, vertex_end)
                chunk = {
                    key: value[start:end] if torch.is_tensor(value) else value
                    for key, value in rt.items()
                }
                chunk_completion = completion[start:end]
                chunk_mask = completion_mask[start:end]
                forced_chunk_ids = forced_ids[start:end]
                target_logits = teacher_logits(
                    teacher, teacher_prompt,
                    {**chunk, "task_ids": forced_chunk_ids},
                    chunk_completion, chunk_mask, use_vertex=True,
                )
                student_logits = completion_logits(
                    model, chunk["prompt_ids"], chunk["prompt_mask"],
                    chunk_completion, chunk_mask,
                    endpoint(
                        prompt_model, forced_chunk_ids,
                        len(forced_chunk_ids), device,
                    ),
                    chunk["positions"],
                )
                chunk_value = full_fkl(target_logits, student_logits, chunk_mask)
                valid_tokens = chunk_mask.sum()
                chunk_weight = (
                    valid_tokens
                    / token_normalizer_by_vertex[historical_task_id]
                    / stage
                )
                (chunk_value * chunk_weight).backward()
                replay_value_sum += chunk_value.detach() * chunk_weight
                task_fkl_numerators[historical_task_id] += (
                    chunk_value.detach().double() * valid_tokens.double()
                )
                task_token_counts[historical_task_id] += valid_tokens.double()
                task_sample_counts[historical_task_id] += end - start
                del target_logits, student_logits
        model.requires_grad_(True)

        allreduce_grads(list(prompt_model.parameters()))
        prompt_clip = clip_prompt_vertex_groups_(
            prompt_model, stage, args.max_grad_norm,
        )
        row_learning_rates = torch.full(
            (len(prompt_model.controls),), args.old_prompt_lr,
            device=device, dtype=torch.float32,
        )
        for historical_task_id in range(stage):
            task_name = TASKS[historical_task_id]
            if task_name in historical_vertex_lr_overrides:
                row_learning_rates[
                    vertex_mask & vertex_task.eq(historical_task_id)
                ] = historical_vertex_lr_overrides[task_name]
        prompt_step_with_mask(
            historical_vertex_mask, args.old_prompt_lr, row_learning_rates,
        )
        prompt_model.requires_grad_(False)

        replay_metric = replay_value_sum.detach().double()
        if dist.is_initialized():
            dist.all_reduce(replay_metric, op=dist.ReduceOp.SUM)
            replay_metric.div_(world)
            dist.all_reduce(task_fkl_numerators, op=dist.ReduceOp.SUM)
            dist.all_reduce(task_token_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(task_sample_counts, op=dist.ReduceOp.SUM)
        replay_fkl_by_task = {
            TASKS[index]: float(
                task_fkl_numerators[index]
                / task_token_counts[index].clamp_min(1)
            )
            for index in range(stage)
        }
        replay_tokens_by_task = {
            TASKS[index]: float(
                task_token_counts[index]
                / task_sample_counts[index].clamp_min(1)
            )
            for index in range(stage)
        }
        if rank == 0:
            print(json.dumps({
                "event": "historical_fkl_step", "phase": phase,
                "stage": stage, "task": task, "slow_step": global_step,
                "transport_step": transport_step,
                "historical_full_fkl": float(replay_metric),
                "historical_fkl_reduction": "equal_vertex_shared_anchor_token_mean",
                "historical_fkl_by_task": replay_fkl_by_task,
                "historical_rollout_tokens_by_task": replay_tokens_by_task,
                "historical_vertices_raw_grad_norm": float(
                    prompt_clip["historical_raw_norm"]
                ),
                "historical_vertices_clip_scale": float(
                    prompt_clip["historical_clip_scale"]
                ),
                "historical_prompt_lr": args.old_prompt_lr,
                "historical_vertex_learning_rates": {
                    TASKS[index]: float(
                        historical_vertex_lr_overrides.get(
                            TASKS[index], args.old_prompt_lr,
                        )
                    )
                    for index in range(stage)
                },
                "replay_global_batch": args.replay_global_batch,
                "shared_anchor_replay_global_batch": 4,
                "shared_anchor_current_global_batch": 4,
                "shared_anchor_applied_to_every_historical_vertex": True,
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "elapsed_s": time.monotonic() - started,
            }), flush=True)
        torch.cuda.reset_peak_memory_stats(device)

    if post_slow_mode:
        post_steps = int(args.post_slow_current_vertex_steps)
        if post_steps > 0:
            model.requires_grad_(False)
            prompt_model.requires_grad_(True)
            for post_step in range(1, post_steps + 1):
                prompt_optimizer.zero_grad(set_to_none=True)
                local_rows = cyclic_rows(
                    current, args.global_batch // world, post_step - 1,
                    rank, world, args.seed + 2999,
                )
                ct = batch_tensors(local_rows, tokenizer, device)
                global_tokens = ct["target_mask"].sum().detach().float()
                if dist.is_initialized():
                    dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
                denominator = global_tokens.clamp_min(1) / world
                nll_sum, token_count = sft_backward(
                    model, prompt_model, ct, stage, args.condition_chunk,
                    denominator,
                )
                allreduce_grads(list(prompt_model.parameters()))
                prompt_clip = clip_prompt_vertex_groups_(
                    prompt_model, stage, args.max_grad_norm,
                )
                lr = args.post_slow_current_vertex_lr * (
                    (post_steps - post_step + 1) / post_steps
                )
                prompt_step_with_mask(current_vertex_mask, lr)
                stats = torch.tensor(
                    [float(nll_sum), float(token_count)], device=device,
                    dtype=torch.float64,
                )
                if dist.is_initialized():
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                if rank == 0:
                    print(json.dumps({
                        "event": "post_slow_current_vertex_step",
                        "stage": stage, "task": task,
                        "step": post_step, "steps": post_steps,
                        "current_vertex_sft_nll": float(
                            stats[0] / stats[1].clamp_min(1)
                        ),
                        "sft_reduction": "global_answer_token_mean",
                        "current_vertex_raw_grad_norm": float(
                            prompt_clip["current_raw_norm"]
                        ),
                        "current_vertex_clip_scale": float(
                            prompt_clip["current_clip_scale"]
                        ),
                        "current_prompt_lr": lr,
                        "slow_frozen": True,
                    }), flush=True)
            prompt_model.requires_grad_(False)
        model.requires_grad_(False)
        prompt_model.requires_grad_(False)
        for _ in range(args.transport_only_steps):
            historical_fkl_update("post_slow_concentrated")
        save_stage(args, model, prompt_model, tokenizer, accelerator, {
            "task": task,
            "epochs": 0,
            "steps": 0,
            "current_vertex_schedule": "external_exact_promot_then_post_slow_recalibration",
            "current_vertex_steps": post_steps,
            "post_slow_current_vertex_lr": args.post_slow_current_vertex_lr,
            "slow_steps": 0,
            "rows": len(current),
            "replay_rows": len(replay),
            "replay_sft_rows": len(replay_sft),
            "historical_fkl_reduction": "equal_vertex_shared_anchor_token_mean",
            "historical_anchor_protocol": "global_8_shared_4_replay_plus_4_current",
            "transport_schedule": "post_slow_concentrated",
            "transport_steps": transport_step,
        })
        if rank == 0:
            print(json.dumps({
                "event": "complete", "stage": stage, "task": task,
                "mode": "post_slow_calibration_and_transport",
                "post_slow_current_vertex_steps": post_steps,
                "transport_steps": transport_step,
                "output": str(args.output_dir),
            }), flush=True)
        return

    # Phase A: fit the newly introduced task vertex against the incoming slow
    # model.  The backbone is strictly frozen, and the masked optimizer restores
    # every historical/interior control row after each step.
    model.requires_grad_(False)
    prompt_model.requires_grad_(True)
    for epoch in range(epochs):
        for local_rows in epoch_batches(
            current, epoch, args.seed, args.global_batch, rank, world,
            steps_per_epoch,
        ):
            if vertex_step >= total_steps:
                break
            vertex_step += 1
            prompt_optimizer.zero_grad(set_to_none=True)
            ct = batch_tensors(local_rows, tokenizer, device)
            current_global_tokens = ct["target_mask"].sum().detach().float()
            if dist.is_initialized():
                dist.all_reduce(current_global_tokens, op=dist.ReduceOp.SUM)
            denominator = current_global_tokens.clamp_min(1) / world
            current_nll_sum, current_token_count = sft_backward(
                model, prompt_model, ct, stage, args.condition_chunk,
                denominator,
            )

            allreduce_grads(list(prompt_model.parameters()))
            prompt_clip = clip_prompt_vertex_groups_(
                prompt_model, stage, args.max_grad_norm,
            )
            current_vertex_lr = current_vertex_learning_rate(vertex_step)
            prompt_step_with_mask(current_vertex_mask, current_vertex_lr)

            vertex_stats = torch.tensor([
                float(current_nll_sum), float(current_token_count),
            ], device=device, dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(vertex_stats, op=dist.ReduceOp.SUM)
            vertex_token_nll = vertex_stats[0] / vertex_stats[1].clamp_min(1)
            if rank == 0:
                print(json.dumps({
                    "event": "current_vertex_step", "stage": stage,
                    "task": task, "step": vertex_step, "steps": total_steps,
                    "epoch": epoch + 1,
                    "current_vertex_sft_nll": float(vertex_token_nll),
                    "sft_reduction": "global_answer_token_mean",
                    "slow_frozen": True,
                    "current_vertex_raw_grad_norm": float(
                        prompt_clip["current_raw_norm"]
                    ),
                    "current_vertex_clip_scale": float(
                        prompt_clip["current_clip_scale"]
                    ),
                    "current_prompt_lr": current_vertex_lr,
                    "current_prompt_lr_scheduler": (
                        args.current_vertex_lr_scheduler_type
                    ),
                    "prompt_control_norm": float(
                        prompt_model.controls.detach().float().norm()
                    ),
                    "peak_memory_gib": (
                        torch.cuda.max_memory_allocated(device) / 2**30
                    ),
                    "elapsed_s": time.monotonic() - started,
                }), flush=True)
            torch.cuda.reset_peak_memory_stats(device)
        if vertex_step >= total_steps:
            break

    # Phase B: the fitted current vertex becomes a fixed coordinate system.
    # Only Slow receives current-task SFT gradients; historical prompt vertices
    # are exposed solely inside historical_fkl_update().
    model.requires_grad_(True)
    prompt_model.requires_grad_(False)

    for epoch in range(epochs):
        for local_rows in epoch_batches(
            current, epoch, args.seed, args.global_batch, rank, world,
            steps_per_epoch,
        ):
            if global_step >= total_steps:
                break
            global_step += 1
            slow_optimizer.zero_grad(set_to_none=True)
            ct = batch_tensors(local_rows, tokenizer, device)
            replay_sft_tensors = None
            replay_sft_importance = 0.0
            if stage > 0 and args.slow_replay_sft and replay_sft:
                replay_sft_rows = cyclic_rows(
                    replay_sft, args.replay_global_batch // world,
                    global_step - 1, rank, world, args.seed + 1999,
                )
                replay_sft_tensors = batch_tensors(replay_sft_rows, tokenizer, device)
                replay_sft_importance = (
                    len(replay_sft) / len(current)
                    * len(local_rows) / len(replay_sft_rows)
                )
            current_global_tokens = ct["target_mask"].sum().detach().float()
            replay_global_tokens = torch.zeros((), device=device)
            if replay_sft_tensors is not None:
                replay_global_tokens = replay_sft_tensors["target_mask"].sum().detach().float()
            if dist.is_initialized():
                dist.all_reduce(current_global_tokens, op=dist.ReduceOp.SUM)
                dist.all_reduce(replay_global_tokens, op=dist.ReduceOp.SUM)
            sft_token_denominator = (
                current_global_tokens
                + replay_sft_importance * replay_global_tokens
            ).clamp_min(1) / world

            current_nll_sum, current_token_count = sft_backward(
                model, prompt_model, ct, stage, args.condition_chunk,
                sft_token_denominator,
            )
            replay_sft_nll_sum = torch.zeros((), device=device)
            replay_sft_token_count = torch.zeros((), device=device)
            if replay_sft_tensors is not None:
                replay_sft_nll_sum, replay_sft_token_count = plain_replay_sft_backward(
                    model, replay_sft_tensors, args.condition_chunk,
                    sft_token_denominator, replay_sft_importance,
                )

            allreduce_grads(model_parameters)
            slow_norm = torch.nn.utils.clip_grad_norm_(model_parameters, args.max_grad_norm)
            slow_optimizer.step()
            slow_scheduler.step()

            sft_stats = torch.tensor([
                float(current_nll_sum), float(current_token_count),
                float(replay_sft_nll_sum), float(replay_sft_token_count),
            ], device=device, dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(sft_stats, op=dist.ReduceOp.SUM)
            current_token_nll = sft_stats[0] / sft_stats[1].clamp_min(1)
            replay_sft_token_nll = sft_stats[2] / sft_stats[3].clamp_min(1)
            if rank == 0:
                print(json.dumps({
                    "event": "slow_step", "stage": stage, "task": task,
                    "step": global_step, "steps": total_steps, "epoch": epoch + 1,
                    "current_sft_nll": float(current_token_nll),
                    "slow_replay_sft_nll": float(replay_sft_token_nll),
                    "sft_reduction": "global_answer_token_mean",
                    "replay_sft_importance": replay_sft_importance,
                    "slow_grad_norm": float(slow_norm),
                    "current_vertex_frozen": True,
                    "slow_lr": slow_scheduler.get_last_lr()[0],
                    "prompt_control_norm": float(prompt_model.controls.detach().float().norm()),
                    "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "elapsed_s": time.monotonic() - started,
                }), flush=True)
            torch.cuda.reset_peak_memory_stats(device)

            if stage > 0 and global_step % args.slow_block_steps == 0:
                for _ in range(args.historical_fkl_steps):
                    historical_fkl_update("periodic")
        if global_step >= total_steps:
            break

    if stage > 0:
        for _ in range(args.final_historical_fkl_steps):
            historical_fkl_update("final")
    save_stage(args, model, prompt_model, tokenizer, accelerator, {
        "task": task, "epochs": epochs, "steps": global_step,
        "current_vertex_schedule": "vertex_then_slow",
        "current_vertex_steps": vertex_step,
        "slow_steps": global_step,
        "rows": len(current), "replay_rows": len(replay),
        "replay_sft_rows": len(replay_sft),
        "current_sft_total_length_dropped": current_sft_dropped,
        "replay_sft_dropped": replay_sft_dropped,
        "sft_reduction": "global_answer_token_mean",
        "historical_fkl_reduction": "equal_task_token_mean",
        "slow_block_steps": args.slow_block_steps,
        "historical_fkl_steps": args.historical_fkl_steps,
        "final_historical_fkl_steps": args.final_historical_fkl_steps,
        "transport_steps": transport_step,
    })
    if rank == 0:
        print(json.dumps({"event": "complete", "stage": stage, "task": task,
                          "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
