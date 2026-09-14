#!/usr/bin/env python3
"""One TRACE stage of OPR-anchored preference SDFT.

Fast memory is a cubic Bezier soft prompt trained at four preferences with
Smooth Tchebycheff scalarisation.  Slow memory is the full backbone, updated by
the two-endpoint MGDA direction.  Acquisition is SDFT Teacher||Student forward
KL on current-task on-policy continuations.  Preservation is ordinary SFT on a
small fixed gold replay buffer.  The first task has no preservation objective.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


from src.preference_memory import (  # noqa: E402
    CubicBezierCode,
    insert_latents,
    padded_prompt_ids,
    prompt_position,
    selective_log_softmax,
    stch_loss,
)
from src.input_conditioned_softprompt import InputConditionedBezierCode  # noqa: E402
from src.task_conditioned_softprompt import TaskConditionedBezierCode  # noqa: E402
from src.distillation import (  # noqa: E402
    teacher_topk_tail_targets,
    topk_tail_forward_kl_from_targets,
)


_CANONICAL_TASKS = (
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
)
TASKS = tuple(filter(None, os.environ.get(
    "TRACE_TASK_ORDER", ",".join(_CANONICAL_TASKS),
).split(",")))
if len(TASKS) != 8 or set(TASKS) != set(_CANONICAL_TASKS):
    raise ValueError(f"invalid TRACE_TASK_ORDER: {TASKS}")
PREFERENCE_COUNT = 4
EPOCHS = {
    "C-STANCE": 5, "FOMC": 3, "MeetingBank": 7, "Py150": 5,
    "ScienceQA": 3, "NumGLUE-cm": 5, "NumGLUE-ds": 5, "20Minuten": 7,
}
MAX_COMPLETION = {
    "C-STANCE": 128, "FOMC": 128, "MeetingBank": 512, "Py150": 128,
    "ScienceQA": 256, "NumGLUE-cm": 128, "NumGLUE-ds": 128,
    "20Minuten": 512,
}
from trace_task_protocol import (  # noqa: E402
    TASK_PROMPTS, ensure_task_prompt, privileged_prompt,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--buffer", type=Path)
    p.add_argument("--previous-prompt", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-train-samples", type=int, default=300)
    p.add_argument("--epochs", type=int)
    p.add_argument("--global-batch", type=int, default=32)
    p.add_argument("--prompt-length", type=int, default=32)
    p.add_argument(
        "--bezier-order", type=int, default=3,
        help="Bezier polynomial order K; the prompt uses K+1 control points.",
    )
    p.add_argument(
        "--fast-init", choices=("zero", "text"), default="zero",
        help=(
            "Initialization used when no previous fast-memory checkpoint is "
            "provided. Text initialization keeps latent tokens on the model's "
            "input-embedding manifold instead of inserting attended zeros."
        ),
    )
    p.add_argument(
        "--fast-init-text",
        default="Answer the question accurately and follow the required output format.",
    )
    p.add_argument("--slow-lr", type=float, default=1e-5)
    p.add_argument("--fast-lr", type=float, default=5e-4)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument(
        "--lr-scheduler-type", choices=("linear", "cosine", "constant"),
        default="linear",
    )
    p.add_argument("--stch-mu", type=float, default=0.1)
    p.add_argument(
        "--fast-scalarization", choices=("stch", "linear"), default="stch",
        help="Smooth Tchebycheff or raw preference-weighted sum for fast memory.",
    )
    p.add_argument(
        "--stch-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply the one-step-delayed batch ideal/range normalization "
            "before Smooth Tchebycheff scalarization.  Disable this to "
            "scalarize the two raw losses directly."
        ),
    )
    p.add_argument("--range-floor", type=float, default=1e-4)
    p.add_argument(
        "--fixed-acquire-scale", type=float,
        help="Fixed functional scale (nats/token) for the acquire objective.",
    )
    p.add_argument(
        "--fixed-preserve-scale", type=float,
        help="Fixed functional scale (nats/token) for the preserve objective.",
    )
    p.add_argument("--ema-alpha", type=float, default=0.01)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-completion-length", type=int)
    p.add_argument("--condition-chunk", type=int, default=3)
    p.add_argument("--generation-chunk", type=int, default=32)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--mgda-fixed-scale-calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Calibrate acquire/preserve gradient norms on the first batch, "
            "then solve MGDA in those fixed normalized units."
        ),
    )
    p.add_argument(
        "--mgda-unit-gradient",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Normalize both endpoint gradients at every step before solving "
            "MGDA; restore the acquire-gradient norm as the common magnitude."
        ),
    )
    p.add_argument(
        "--slow-gradient-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use acquisition as the primary slow objective and project away "
            "only its component that would increase the preserve loss."
        ),
    )
    p.add_argument(
        "--slow-dwa", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Use Dynamic Weight Averaging on fixed-scale-normalized slow "
            "endpoint gradients instead of the min-norm MGDA coefficient."
        ),
    )
    p.add_argument("--dwa-temperature", type=float, default=2.0)
    p.add_argument("--dwa-window", type=int, default=10)
    p.add_argument(
        "--training-schedule",
        choices=(
            "task_twophase", "joint_random_then_fast", "baseline_then_fast",
            "fast_only", "joint_onephase", "uniform_then_fast",
        ),
        default="task_twophase",
        help=(
            "task_twophase trains a plain slow phase then a fast STCH phase. "
            "joint_random_then_fast first updates theta and phi jointly at "
            "one batch-shared random preference using DWA/projection, then "
            "freezes theta and refines phi with four-preference STCH. "
            "baseline_then_fast exactly trains slow on the concatenation of "
            "current data and replay with ordinary SFT, then freezes slow and "
            "trains only the four-preference fast STCH path. fast_only loads "
            "an externally trained slow checkpoint and prompt initializer, "
            "then runs only the latter STCH phase."
            " uniform_then_fast first freezes fast memory and trains slow at "
            "lambda=0.5, then freezes slow and trains fast at four random "
            "preferences without forced endpoints."
        ),
    )
    p.add_argument(
        "--onephase-slow-mode", choices=("acquire_sft", "projection"),
        default="projection",
        help=(
            "For joint_onephase, update slow with current-task plain SFT only "
            "or with acquire-primary projection against replay."
        ),
    )
    p.add_argument(
        "--input-conditioned-fast", action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate z(x, lambda) with a frozen text encoder and low-rank hypernetwork.",
    )
    p.add_argument(
        "--task-conditioned-fast", action=argparse.BooleanOptionalAction,
        default=False,
        help="Use one oracle-task-conditioned cubic Bezier prompt path per TRACE task.",
    )
    p.add_argument(
        "--preserve-objective", choices=("sft", "topk_fkl"), default="sft",
        help="Replay objective for fast memory.",
    )
    p.add_argument(
        "--preserve-model", type=Path,
        help="Frozen previous-stage checkpoint used by top-k+OTHER preserve FKL.",
    )
    p.add_argument("--preserve-top-k", type=int, default=64)
    p.add_argument(
        "--endpoint-transport", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Continue the complete previous Bezier path: initialize each "
            "current control P_t[k] from P_{t-1}[k], and condition the frozen "
            "preserve teacher with z_{t-1}(lambda) at matching lambda."
        ),
    )
    p.add_argument(
        "--save-frozen-model", action=argparse.BooleanOptionalAction,
        default=True,
        help="Save another copy of the frozen slow model in fast-only runs.",
    )
    p.add_argument("--condition-width", type=int, default=512)
    p.add_argument("--condition-rank", type=int, default=8)
    p.add_argument("--max-question-encoder-length", type=int, default=384)
    p.add_argument(
        "--condition-acquire-gate", action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Multiply the question-conditioned residual by (1-lambda). "
            "Disable to give all preferences equal conditional capacity."
        ),
    )
    p.add_argument("--acquire-kl-direction", choices=("forward", "reverse"), default="forward")
    p.add_argument(
        "--acquire-objective", choices=("sdft", "sft"), default="sdft",
        help="Use on-policy self-distillation KL or gold-response SFT NLL for acquisition.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--endpoint-forward-reuse", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse pre-fast-update endpoint graphs. This is faster but is not "
            "equivalent to the original fast-then-slow optimization order."
        ),
    )
    p.add_argument(
        "--slow-four-preference-mgda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Jointly update the slow backbone with the min-norm MGDA "
            "direction of four batch-shared preference losses while the "
            "fast prompt follows their mean STCH loss. Both updates reuse "
            "the same forward graphs."
        ),
    )
    p.add_argument(
        "--slow-four-preference-common-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Update the slow backbone with the component of the mean "
            "four-preference gradient orthogonal to all centered preference "
            "differences. The singular centered Gram matrix is solved with "
            "a Moore-Penrose pseudoinverse; no diagonal epsilon is added."
        ),
    )
    p.add_argument(
        "--slow-endpoint-safe-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse the lambda=0 acquire and lambda=1 preserve endpoint "
            "forwards: update slow with the acquire gradient projected onto "
            "the preserve-safe half-space (beta=0), while all four STCH "
            "objectives update only the fast prompt."
        ),
    )
    p.add_argument(
        "--batch-shared-random-preferences",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use four sorted U(0,1) preferences shared by the complete "
            "distributed batch, without forcing either endpoint."
        ),
    )
    p.add_argument("--length-bucket-size", type=int, default=128)
    return p.parse_args()


def render(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def read_rows(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def prepare_current_rows(args, tokenizer):
    raw = read_rows(args.data_root / args.task / "train.json")
    candidates = []
    for index, row in enumerate(raw):
        task_prompt = ensure_task_prompt(args.task, str(row["prompt"]))
        prompt = render(tokenizer, task_prompt)
        teacher = render(tokenizer, privileged_prompt(
            args.task, task_prompt, str(row["answer"]),
        ))
        candidates.append((index, row, task_prompt, prompt, teacher))
    # One Rust-tokenizer batch call is substantially faster than 10k Python
    # calls and is reused throughout all epochs.
    prompt_batch = tokenizer(
        [item[3] for item in candidates], add_special_tokens=False,
        padding=False, truncation=False,
    ).input_ids
    teacher_batch = tokenizer(
        [item[4] for item in candidates], add_special_tokens=False,
        padding=False, truncation=False,
    ).input_ids
    answer_batch = tokenizer(
        [str(item[1]["answer"]) for item in candidates],
        add_special_tokens=False, padding=False, truncation=True,
        max_length=max(1, args.max_completion_length - 1),
    ).input_ids
    kept = []
    for (index, row, task_prompt, prompt, teacher), prompt_ids, teacher_ids, answer_ids in zip(
        candidates, prompt_batch, teacher_batch, answer_batch,
    ):
        required_prompt_length = (
            max(len(prompt_ids), len(teacher_ids))
            if args.acquire_objective == "sdft" else len(prompt_ids)
        )
        if required_prompt_length <= args.max_prompt_length:
            if not answer_ids or answer_ids[-1] != tokenizer.eos_token_id:
                answer_ids.append(tokenizer.eos_token_id)
            answer_ids = answer_ids[:args.max_completion_length]
            kept.append({
                "prompt": task_prompt, "answer": str(row["answer"]),
                "task": args.task, "task_id": TASKS.index(args.task),
                "prompt_text": prompt, "teacher_text": teacher,
                "prompt_ids": prompt_ids, "teacher_ids": teacher_ids,
                "answer_ids": answer_ids,
                "length": required_prompt_length + len(answer_ids),
                "source_index": index,
            })
    random.Random(args.seed).shuffle(kept)
    return kept[:min(args.max_train_samples, len(kept))], len(raw) - len(kept)


def prepare_buffer_rows(args, tokenizer):
    if args.stage == 0 or args.buffer is None or not args.buffer.exists():
        return []
    rows = read_rows(args.buffer)
    manifest_path = args.buffer.with_name("buffer.manifest.json")
    if rows and all("task" not in row for row in rows) and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        allocation = manifest.get("allocation", {})
        if sum(int(count) for count in allocation.values()) != len(rows):
            raise ValueError(
                f"Gold replay manifest allocation does not match buffer size: "
                f"{manifest_path}"
            )
        offset = 0
        for source_task, count in allocation.items():
            for row in rows[offset:offset + int(count)]:
                row["_source_task"] = source_task
            offset += int(count)
    candidates = []
    for row in rows:
        raw_prompt = str(row["prompt"])
        if "task" in row or "_source_task" in row:
            row_task = str(row.get("task", row.get("_source_task")))
        else:
            matches = [
                task for task, prefix in TASK_PROMPTS.items()
                if raw_prompt.startswith(prefix)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Cannot infer replay task from canonical prompt prefix; "
                    f"matches={matches}, prompt={raw_prompt[:120]!r}"
                )
            row_task = matches[0]
        prompt_text = ensure_task_prompt(row_task, raw_prompt)
        prompt = render(tokenizer, prompt_text)
        answer = str(row["answer"])
        candidates.append((row, row_task, prompt, answer))
    prompt_batch = tokenizer(
        [item[2] for item in candidates], add_special_tokens=False,
        padding=False, truncation=False,
    ).input_ids
    answer_batch = tokenizer(
        [item[3] for item in candidates], add_special_tokens=False,
        padding=False, truncation=True,
        max_length=max(1, args.max_completion_length - 1),
    ).input_ids
    result = []
    for (row, row_task, prompt, answer), prompt_ids, answer_ids in zip(
        candidates, prompt_batch, answer_batch,
    ):
        if not answer_ids or answer_ids[-1] != tokenizer.eos_token_id:
            answer_ids.append(tokenizer.eos_token_id)
        answer_ids = answer_ids[:args.max_completion_length]
        original_prompt_length = len(prompt_ids)
        replay_overlength = getattr(args, "replay_overlength", "drop")
        if original_prompt_length > args.max_prompt_length:
            if replay_overlength != "head_tail":
                continue
            # Preserve both the canonical task instruction at the beginning
            # and the concrete problem plus assistant boundary at the end.
            head = min(256, args.max_prompt_length // 4)
            tail = args.max_prompt_length - head
            prompt_ids = prompt_ids[:head] + prompt_ids[-tail:]
        result.append({
            **row, "prompt_text": prompt, "answer": answer,
            "task": row_task, "task_id": TASKS.index(row_task),
            "prompt_ids": prompt_ids, "answer_ids": answer_ids,
            "length": len(prompt_ids) + len(answer_ids),
            "prompt_truncated": original_prompt_length > len(prompt_ids),
            "original_prompt_length": original_prompt_length,
        })
    return result


def prepare_baseline_sft_rows(rows, tokenizer, max_length=2048):
    """Retokenize SFT exactly like the plain PyTorch SFT+Replay baseline.

    The baseline truncates the concatenated ``prompt + answer + EOS`` to one
    total sequence length.  Rollout/FKL preprocessing remains task-specific
    and must not be reused as the supervised SFT token protocol.
    """
    if not rows:
        return [], 0
    prompt_batch = tokenizer(
        [row["prompt_text"] for row in rows], add_special_tokens=False,
        padding=False, truncation=False,
    ).input_ids
    answer_batch = tokenizer(
        [str(row["answer"]) for row in rows], add_special_tokens=False,
        padding=False, truncation=False,
    ).input_ids
    result = []
    for row, prompt_ids, answer_ids in zip(rows, prompt_batch, answer_batch):
        answer_with_eos = [*answer_ids, tokenizer.eos_token_id]
        available = int(max_length) - len(prompt_ids)
        if available <= 0:
            continue
        target_ids = answer_with_eos[:available]
        if not target_ids:
            continue
        result.append({
            **row,
            "prompt_ids": prompt_ids,
            "answer_ids": target_ids,
            "length": len(prompt_ids) + len(target_ids),
            "plain_sft_total_length": int(max_length),
        })
    return result, len(rows) - len(result)


def pad_id_rows(values, pad_id, device, padding_side="left"):
    width = max(len(ids) for ids in values)
    tensor = torch.full(
        (len(values), width), pad_id, dtype=torch.long, device=device,
    )
    mask = torch.zeros_like(tensor)
    positions = torch.zeros(len(values), dtype=torch.long, device=device)
    for index, ids in enumerate(values):
        item = torch.tensor(ids, dtype=torch.long, device=device)
        if padding_side == "left":
            tensor[index, width - len(ids):] = item
            mask[index, width - len(ids):] = 1
            positions[index] = width
        else:
            tensor[index, :len(ids)] = item
            mask[index, :len(ids)] = 1
    return tensor, mask, positions


def preference_values(question_count, step, seed, device, rank, world,
                      batch_shared=False, symmetric_endpoints=False):
    """Sample four preferences, optionally shared by the global batch."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        seed + 104729 * step + (0 if batch_shared else 1009 * rank)
    )
    if batch_shared:
        # A preference index must denote the same objective on every example
        # and every rank before it is meaningful to apply multi-objective
        # gradient geometry to the four indexed gradients.
        if symmetric_endpoints:
            value = torch.rand((), generator=generator) * 0.5
            shared = torch.stack((
                torch.zeros(()), value, 1.0 - value, torch.ones(()),
            ))
        else:
            shared = torch.rand(4, generator=generator).clamp_(
                1e-6, 1.0 - 1e-6,
            ).sort().values
        return shared.unsqueeze(0).expand(question_count, -1).reshape(-1).to(device)
    random_values = torch.rand(
        question_count, 2, generator=generator,
    ).clamp_(1e-6, 1 - 1e-6)
    values = torch.stack((
        torch.zeros(question_count),
        random_values[:, 0],
        random_values[:, 1],
        torch.ones(question_count),
    ), dim=1)
    return values.reshape(-1).to(device)


def completion_tensors(tokenizer, texts, max_length, device):
    values = []
    for text in texts:
        ids = tokenizer(
            text, add_special_tokens=False, truncation=True,
            max_length=max(1, max_length - 1),
        ).input_ids
        if not ids or ids[-1] != tokenizer.eos_token_id:
            ids.append(tokenizer.eos_token_id)
        values.append(ids[:max_length])
    width = max(len(ids) for ids in values)
    tensor = torch.full(
        (len(values), width), tokenizer.pad_token_id,
        dtype=torch.long, device=device,
    )
    mask = torch.zeros_like(tensor)
    for i, ids in enumerate(values):
        tensor[i, :len(ids)] = torch.tensor(ids, device=device)
        mask[i, :len(ids)] = 1
    return tensor, mask


def prompt_tensors(tokenizer, texts, max_length, device):
    return padded_prompt_ids(tokenizer, texts, max_length, device)[:3]


def completion_logits(model, prompt_ids, prompt_mask, completion_ids,
                      completion_mask, code, positions):
    content_ids = torch.cat((prompt_ids, completion_ids), dim=1)
    content_mask = torch.cat((prompt_mask, completion_mask), dim=1)
    embeds, expanded_mask = insert_latents(
        model.get_input_embeddings(), content_ids, content_mask, code, positions,
    )
    length = completion_ids.shape[1]
    return model(
        inputs_embeds=embeds, attention_mask=expanded_mask,
        use_cache=False, return_dict=True, logits_to_keep=length + 1,
    ).logits[:, -(length + 1):-1]


def plain_completion_logits(model, prompt_ids, prompt_mask, completion_ids,
                            completion_mask):
    """Causal-LM logits without inserting any fast-memory tokens."""
    content_ids = torch.cat((prompt_ids, completion_ids), dim=1)
    content_mask = torch.cat((prompt_mask, completion_mask), dim=1)
    length = completion_ids.shape[1]
    return model(
        input_ids=content_ids, attention_mask=content_mask,
        use_cache=False, return_dict=True, logits_to_keep=length + 1,
    ).logits[:, -(length + 1):-1]


@torch.no_grad()
def teacher_completion_logits(teacher, teacher_ids, teacher_mask,
                              completion_ids, completion_mask):
    content_ids = torch.cat((teacher_ids, completion_ids), dim=1)
    content_mask = torch.cat((teacher_mask, completion_mask), dim=1)
    length = completion_ids.shape[1]
    return teacher(
        input_ids=content_ids, attention_mask=content_mask,
        use_cache=False, return_dict=True, logits_to_keep=length + 1,
    ).logits[:, -(length + 1):-1]


def forward_kl_per_example(teacher_logits, student_logits, mask):
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    token_kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)
    return (token_kl * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def reverse_kl_per_example(teacher_logits, student_logits, mask):
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    token_kl = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1)
    return (token_kl * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def acquire_kl_per_example(teacher_logits, student_logits, mask, args):
    function = forward_kl_per_example if args.acquire_kl_direction == "forward" else reverse_kl_per_example
    return function(teacher_logits, student_logits, mask)


def sft_nll_per_example(logits, targets, mask):
    token_logp = selective_log_softmax(logits, targets).float()
    return -(token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def stage0_plain_sft_update(model, tensors, optimizer, scheduler, args):
    """Exact first-task degeneration: ordinary SFT with no fast memory.

    ``build_step_tensors`` repeats every question for the four preference
    values used by later stages.  Select one copy per question here so stage 0
    has exactly the requested global batch and exactly the plain SFT graph.
    """
    model.requires_grad_(True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    indices = torch.arange(
        0, len(tensors["lambdas"]), tensors.get("preference_count", PREFERENCE_COUNT),
        device=tensors["prompt_ids"].device,
    )
    question_count = len(indices)
    loss_sum = 0.0
    token_sum = 0.0
    for start in range(0, question_count, args.condition_chunk):
        chosen = indices[start:start + args.condition_chunk]
        logits = plain_completion_logits(
            model,
            tensors["prompt_ids"][chosen], tensors["prompt_mask"][chosen],
            tensors["current_target_ids"][chosen],
            tensors["current_target_mask"][chosen],
        )
        losses = sft_nll_per_example(
            logits, tensors["current_target_ids"][chosen],
            tensors["current_target_mask"][chosen],
        )
        (losses.sum() / question_count).backward()
        loss_sum += float(losses.detach().sum())
        token_sum += float(tensors["current_target_mask"][chosen].sum())
        del logits, losses
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    fused_allreduce_parameter_grads(parameters, dist.get_world_size())
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    mean_loss = loss_sum / question_count
    return {
        "fast_loss": None,
        "acquire_fkl": mean_loss,
        "preserve_sft_nll": None,
        "fast_effective_acquire_weight": None,
        "stch_entropy": None,
        "stch_min": None,
        "stch_max": None,
        "stch_range": None,
        "normalization_delay_steps": None,
        "fast_grad_norm": None,
        "slow_acquire_fkl_lambda0": mean_loss,
        "slow_preserve_sft_nll_lambda1": None,
        "mgda_alpha_acquire": 1.0,
        "grad_norm_acquire": float(grad_norm),
        "grad_norm_preserve": 0.0,
        "gradient_cosine": 0.0,
        "mgda_calibration_acquire_scale": None,
        "mgda_calibration_preserve_scale": None,
        "calibrated_grad_norm_acquire": None,
        "calibrated_grad_norm_preserve": None,
        "mgda_common_scale": None,
        "projection_active": None,
        "projection_removed_ratio": None,
        "preserve_dot_after_projection": None,
        "slow_grad_norm_preclip": float(grad_norm),
        "slow_lr": scheduler.get_last_lr()[0],
        "endpoint_forward_reused": False,
        "stage0_plain_sft": True,
        "plain_target_tokens_mean": token_sum / question_count,
    }


def backward_plain_sft_endpoint(model, prompt_ids, prompt_mask, target_ids,
                                target_mask, indices, chunk_size):
    """Backpropagate a no-fast-memory SFT objective over unique questions."""
    losses = []
    denominator = len(indices)
    for start in range(0, denominator, chunk_size):
        chosen = indices[start:start + chunk_size]
        logits = plain_completion_logits(
            model, prompt_ids[chosen], prompt_mask[chosen],
            target_ids[chosen], target_mask[chosen],
        )
        value = sft_nll_per_example(
            logits, target_ids[chosen], target_mask[chosen],
        )
        (value.sum() / denominator).backward()
        losses.append(value.detach())
        del logits, value
    return torch.cat(losses).mean()


def slow_plain_projection_update(model, tensors, optimizer, scheduler, args,
                                 mgda_calibration_state, dwa_state):
    """Update slow weights from plain acquire/preserve SFT objectives.

    Neither endpoint contains a soft prompt.  Fast memory therefore cannot
    become a moving coordinate system for the backbone update.  Geometry is
    either acquire-primary projection or genuine two-objective MGDA.
    """
    model.requires_grad_(True)
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    indices = torch.arange(
        0, len(tensors["lambdas"]), PREFERENCE_COUNT,
        device=tensors["prompt_ids"].device,
    )

    optimizer.zero_grad(set_to_none=True)
    acquire = backward_plain_sft_endpoint(
        model,
        tensors["prompt_ids"], tensors["prompt_mask"],
        tensors["current_target_ids"], tensors["current_target_mask"],
        indices, args.condition_chunk,
    )
    flat_a = fused_allreduce_parameter_grads(
        parameters, dist.get_world_size(),
    ).detach().clone()

    optimizer.zero_grad(set_to_none=True)
    preserve = backward_plain_sft_endpoint(
        model,
        tensors["replay_ids"], tensors["replay_mask"],
        tensors["target_ids"], tensors["target_mask"],
        indices, args.condition_chunk,
    )
    flat_p = fused_allreduce_parameter_grads(parameters, dist.get_world_size())
    projection = geometry = None
    if args.slow_dwa:
        geometry = dwa_direction(
            flat_a, flat_p, mgda_calibration_state, dwa_state,
            float(acquire), float(preserve),
            args.dwa_temperature, args.dwa_window,
        )
        result = geometry
        alpha = None
    elif args.slow_gradient_projection:
        projection = acquire_primary_projection(flat_a, flat_p)
        result = projection
        alpha = None
    else:
        geometry = calibrated_mgda(
            flat_a, flat_p, mgda_calibration_state,
            args.mgda_fixed_scale_calibration, args.mgda_unit_gradient,
        )
        result = geometry
        alpha = geometry["alpha"]
    combined = torch._utils._unflatten_dense_tensors(
        result["combined"], parameters,
    )
    for parameter, gradient in zip(parameters, combined):
        parameter.grad = gradient
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    norm_a = result["raw_norm_a_sq"]
    norm_p = result["raw_norm_p_sq"]
    dot = result["raw_dot"]
    return {
        "slow_acquire_fkl_lambda0": float(acquire),
        "slow_preserve_sft_nll_lambda1": float(preserve),
        "mgda_alpha_acquire": alpha,
        "grad_norm_acquire": math.sqrt(max(norm_a, 0.0)),
        "grad_norm_preserve": math.sqrt(max(norm_p, 0.0)),
        "gradient_cosine": (
            0.0 if norm_a <= 0 or norm_p <= 0
            else dot / math.sqrt(norm_a * norm_p)
        ),
        "mgda_calibration_acquire_scale": (
            None if geometry is None else geometry["scale_a"]
        ),
        "mgda_calibration_preserve_scale": (
            None if geometry is None else geometry["scale_p"]
        ),
        "calibrated_grad_norm_acquire": (
            None if geometry is None else geometry["calibrated_norm_a"]
        ),
        "calibrated_grad_norm_preserve": (
            None if geometry is None else geometry["calibrated_norm_p"]
        ),
        "mgda_common_scale": (
            None if geometry is None else geometry["common_scale"]
        ),
        "dwa_weight_acquire": (
            None if not args.slow_dwa else geometry["dwa_weight_acquire"]
        ),
        "dwa_weight_preserve": (
            None if not args.slow_dwa else geometry["dwa_weight_preserve"]
        ),
        "dwa_ratio_acquire": (
            None if not args.slow_dwa else geometry["dwa_ratio_acquire"]
        ),
        "dwa_ratio_preserve": (
            None if not args.slow_dwa else geometry["dwa_ratio_preserve"]
        ),
        "projection_active": None if projection is None else projection["active"],
        "projection_removed_ratio": (
            None if projection is None else projection["removed_ratio"]
        ),
        "preserve_dot_after_projection": (
            None if projection is None else projection["dot_after"]
        ),
        "slow_grad_norm_preclip": float(grad_norm),
        "slow_lr": scheduler.get_last_lr()[0],
        "slow_prompt_conditioning": "none",
    }


@torch.no_grad()
def generate_current(model, code_model, prompt_ids, prompt_mask, positions,
                     lambdas, tokenizer, args):
    outputs, masks = [], []
    old_cache = model.config.use_cache
    model.config.use_cache = True
    model.eval()
    for start in range(0, len(lambdas), args.generation_chunk):
        end = min(start + args.generation_chunk, len(lambdas))
        code = code_model(lambdas[start:end]).detach()
        embeds, expanded_mask = insert_latents(
            model.get_input_embeddings(), prompt_ids[start:end],
            prompt_mask[start:end], code, positions[start:end],
        )
        generated = model.generate(
            inputs_embeds=embeds, attention_mask=expanded_mask,
            do_sample=True, temperature=1.0, top_p=1.0,
            max_new_tokens=args.max_completion_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, use_cache=True,
        )
        mask = torch.ones_like(generated, dtype=torch.long)
        for row_index, values in enumerate(generated):
            eos = values.eq(tokenizer.eos_token_id).nonzero().flatten()
            if eos.numel():
                mask[row_index, int(eos[0]) + 1:] = 0
        outputs.append(generated)
        masks.append(mask)
    model.config.use_cache = old_cache
    width = max(value.shape[1] for value in outputs)
    padded, padded_masks = [], []
    for value, mask in zip(outputs, masks):
        if value.shape[1] < width:
            amount = width - value.shape[1]
            value = F.pad(value, (0, amount), value=tokenizer.pad_token_id)
            mask = F.pad(mask, (0, amount), value=0)
        padded.append(value)
        padded_masks.append(mask)
    return torch.cat(padded), torch.cat(padded_masks)


def build_step_tensors(rows, replay_rows, lambdas, tokenizer, args, device):
    if len(lambdas) % len(rows):
        raise ValueError("lambda count must be an integer multiple of row count")
    pref_count = len(lambdas) // len(rows)
    current_ids = [row["prompt_ids"] for row in rows for _ in range(pref_count)]
    # Replay rows do not carry privileged teacher prompts.  Plain SFT does
    # not consume this field, so using the ordinary prompt as a shape-safe
    # placeholder keeps the mixed baseline data path free of extra tokenization.
    teacher_token_ids = [
        row.get("teacher_ids", row["prompt_ids"])
        for row in rows for _ in range(pref_count)
    ]
    current_answer_ids = [row["answer_ids"] for row in rows for _ in range(pref_count)]
    prompt_ids, prompt_mask, positions = pad_id_rows(
        current_ids, tokenizer.pad_token_id, device,
    )
    # Soft memory is always inserted immediately before the assistant turn.
    # This is the coordinate used by the source-stage trainer and evaluator;
    # inserting transported controls at the end changes their function.
    prompt_width = prompt_ids.shape[1]
    positions = torch.tensor([
        prompt_width - len(ids) + prompt_position(tokenizer, ids)
        for ids in current_ids
    ], dtype=torch.long, device=device)
    teacher_ids, teacher_mask, _ = pad_id_rows(
        teacher_token_ids, tokenizer.pad_token_id, device,
    )
    current_target_ids, current_target_mask, _ = pad_id_rows(
        current_answer_ids, tokenizer.pad_token_id, device, padding_side="right",
    )
    result = {
        "prompt_ids": prompt_ids, "prompt_mask": prompt_mask,
        "positions": positions, "teacher_ids": teacher_ids,
        "teacher_mask": teacher_mask,
        "current_target_ids": current_target_ids,
        "current_target_mask": current_target_mask,
        "lambdas": lambdas,
        "preference_count": pref_count,
        "task_ids": torch.tensor(
            [row["task_id"] for row in rows for _ in range(pref_count)],
            dtype=torch.long, device=device,
        ),
    }
    if replay_rows:
        replay_token_ids = [row["prompt_ids"] for row in replay_rows for _ in range(pref_count)]
        replay_answer_ids = [row["answer_ids"] for row in replay_rows for _ in range(pref_count)]
        replay_ids, replay_mask, replay_positions = pad_id_rows(
            replay_token_ids, tokenizer.pad_token_id, device,
        )
        replay_width = replay_ids.shape[1]
        replay_positions = torch.tensor([
            replay_width - len(ids) + prompt_position(tokenizer, ids)
            for ids in replay_token_ids
        ], dtype=torch.long, device=device)
        target_ids, target_mask, _ = pad_id_rows(
            replay_answer_ids, tokenizer.pad_token_id, device, padding_side="right",
        )
        result.update({
            "replay_ids": replay_ids, "replay_mask": replay_mask,
            "replay_positions": replay_positions,
            "target_ids": target_ids, "target_mask": target_mask,
            "replay_task_ids": torch.tensor(
                [row["task_id"] for row in replay_rows for _ in range(pref_count)],
                dtype=torch.long, device=device,
            ),
        })
    return result


def prompt_code(code_model, lambdas, tensors, replay=False):
    """Dispatch global, x-conditioned, and oracle-task-conditioned prompts."""
    if getattr(code_model, "task_conditioned", False):
        key = "replay_task_ids" if replay else "task_ids"
        return code_model(lambdas, tensors[key])
    feature_key = "replay_question_features" if replay else "question_features"
    if feature_key in tensors:
        return code_model(lambdas, tensors[feature_key])
    return code_model(lambdas)


def objective_chunks(model, teacher, code_model, tensors, completions,
                     completion_mask, args, need_preserve=True,
                     preserve_teacher=None, transport_source_prompt=None):
    acquire_parts, preserve_parts = [], []
    count = len(tensors["lambdas"])
    for start in range(0, count, args.condition_chunk):
        end = min(start + args.condition_chunk, count)
        chunk_lambdas = tensors["lambdas"][start:end]
        code = prompt_code(code_model, chunk_lambdas, tensors)
        if args.acquire_objective == "sft":
            student_logits = completion_logits(
                model, tensors["prompt_ids"][start:end],
                tensors["prompt_mask"][start:end],
                tensors["current_target_ids"][start:end],
                tensors["current_target_mask"][start:end], code,
                tensors["positions"][start:end],
            )
            acquire_parts.append(sft_nll_per_example(
                student_logits, tensors["current_target_ids"][start:end],
                tensors["current_target_mask"][start:end],
            ))
            del student_logits
        else:
            teacher_logits = teacher_completion_logits(
                teacher, tensors["teacher_ids"][start:end],
                tensors["teacher_mask"][start:end], completions[start:end],
                completion_mask[start:end],
            )
            student_logits = completion_logits(
                model, tensors["prompt_ids"][start:end],
                tensors["prompt_mask"][start:end], completions[start:end],
                completion_mask[start:end], code, tensors["positions"][start:end],
            )
            acquire_parts.append(acquire_kl_per_example(
                teacher_logits, student_logits, completion_mask[start:end], args,
            ))
            del teacher_logits, student_logits
        if need_preserve:
            replay_code = prompt_code(code_model, chunk_lambdas, tensors, replay=True)
            replay_logits = completion_logits(
                model, tensors["replay_ids"][start:end],
                tensors["replay_mask"][start:end],
                tensors["target_ids"][start:end],
                tensors["target_mask"][start:end], replay_code,
                tensors["replay_positions"][start:end],
            )
            if args.preserve_objective == "topk_fkl":
                if preserve_teacher is None:
                    raise RuntimeError("topk_fkl requires a frozen preserve teacher")
                if "preserve_top_idx" in tensors:
                    top_idx = tensors["preserve_top_idx"][start:end]
                    top_logp = tensors["preserve_top_logp"][start:end]
                    top_mass = tensors["preserve_top_mass"][start:end]
                    old_logits = None
                else:
                  with torch.no_grad():
                    if transport_source_prompt is None:
                        old_logits = plain_completion_logits(
                            preserve_teacher,
                            tensors["replay_ids"][start:end],
                            tensors["replay_mask"][start:end],
                            tensors["target_ids"][start:end],
                            tensors["target_mask"][start:end],
                        )
                    else:
                        if transport_source_prompt.ndim == 3:
                            teacher_basis = CubicBezierCode.basis(
                                chunk_lambdas.float()
                            ).to(transport_source_prompt.dtype)
                            teacher_code = torch.einsum(
                                "bk,kmh->bmh",
                                teacher_basis, transport_source_prompt,
                            )
                        else:
                            teacher_code = transport_source_prompt.unsqueeze(0).expand(
                                end - start, -1, -1,
                            )
                        old_logits = completion_logits(
                            preserve_teacher,
                            tensors["replay_ids"][start:end],
                            tensors["replay_mask"][start:end],
                            tensors["target_ids"][start:end],
                            tensors["target_mask"][start:end],
                            teacher_code,
                            tensors["replay_positions"][start:end],
                        )
                    top_idx, top_logp, top_mass = teacher_topk_tail_targets(
                        old_logits, top_k=args.preserve_top_k,
                    )
                token_kl, _, _ = topk_tail_forward_kl_from_targets(
                    top_idx, top_logp, top_mass, replay_logits,
                )
                mask = tensors["target_mask"][start:end]
                preserve_parts.append(
                    (token_kl * mask).sum(-1) / mask.sum(-1).clamp_min(1)
                )
                del top_idx, top_logp, top_mass, token_kl
                if old_logits is not None:
                    del old_logits
            else:
                preserve_parts.append(sft_nll_per_example(
                    replay_logits, tensors["target_ids"][start:end],
                    tensors["target_mask"][start:end],
                ))
            del replay_logits, replay_code
    return torch.cat(acquire_parts), (
        torch.cat(preserve_parts) if preserve_parts else None
    )


@torch.no_grad()
def attach_constant_preserve_targets(
    tensors, preserve_teacher, transport_source_prompt, args,
):
    """Score each replay question once when the frozen old path is constant.

    A repeated Bezier control represents exactly the same old policy at every
    preference.  Repeating its compressed target is mathematically identical
    to rescoring all four copies and removes 75% of preserve-teacher forwards.
    """
    if (preserve_teacher is None or transport_source_prompt is None
            or "replay_ids" not in tensors):
        return False
    controls = transport_source_prompt
    if controls.ndim != 3 or not torch.equal(
        controls, controls[:1].expand_as(controls)
    ):
        return False
    pref_count = int(tensors.get("preference_count", PREFERENCE_COUNT))
    chosen = torch.arange(
        0, len(tensors["lambdas"]), pref_count,
        device=tensors["lambdas"].device,
    )
    top_idx_parts, top_logp_parts, top_mass_parts = [], [], []
    code = controls[0].unsqueeze(0)
    for start in range(0, len(chosen), args.condition_chunk):
        indices = chosen[start:start + args.condition_chunk]
        old_logits = completion_logits(
            preserve_teacher,
            tensors["replay_ids"][indices], tensors["replay_mask"][indices],
            tensors["target_ids"][indices], tensors["target_mask"][indices],
            code.expand(len(indices), -1, -1),
            tensors["replay_positions"][indices],
        )
        top_idx, top_logp, top_mass = teacher_topk_tail_targets(
            old_logits, top_k=args.preserve_top_k,
        )
        top_idx_parts.append(top_idx)
        top_logp_parts.append(top_logp)
        top_mass_parts.append(top_mass)
    tensors["preserve_top_idx"] = torch.cat(top_idx_parts).repeat_interleave(
        pref_count, dim=0,
    )
    tensors["preserve_top_logp"] = torch.cat(top_logp_parts).repeat_interleave(
        pref_count, dim=0,
    )
    tensors["preserve_top_mass"] = torch.cat(top_mass_parts).repeat_interleave(
        pref_count, dim=0,
    )
    return True


def slice_tensors(tensors, start, end):
    """Slice every sequence-major tensor while retaining scalar metadata."""
    count = len(tensors["lambdas"])
    return {
        key: (
            value[start:end]
            if torch.is_tensor(value) and value.ndim and value.shape[0] == count
            else value
        )
        for key, value in tensors.items()
    }


def fused_allreduce_parameter_grads(parameters, divisor):
    """One coalesced collective instead of one all-reduce per parameter."""
    gradients = [
        parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        for parameter in parameters
    ]
    flat = torch._utils._flatten_dense_tensors(gradients)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(divisor)
    reduced = torch._utils._unflatten_dense_tensors(flat, gradients)
    for parameter, gradient in zip(parameters, reduced):
        parameter.grad = gradient
    return flat


def flat_norm_dot(first, second=None, chunk_elements=16_777_216):
    """FP32 gradient geometry without materializing model-sized FP32 copies."""
    total = 0.0
    for start in range(0, first.numel(), chunk_elements):
        left = first[start:start + chunk_elements].float()
        right = left if second is None else second[start:start + chunk_elements].float()
        total += float(torch.dot(left, right))
    return total


def calibrated_mgda(flat_a, flat_p, calibration_state, fixed_scale, unit_gradient):
    """Return a fixed-scale calibrated two-objective MGDA direction.

    Calibration is initialized exactly once from the first distributed batch.
    The geometric-mean multiplier restores a common gradient scale without
    changing the calibrated MGDA direction.
    """
    raw_norm_a_sq = flat_norm_dot(flat_a)
    raw_norm_p_sq = flat_norm_dot(flat_p)
    raw_dot = flat_norm_dot(flat_a, flat_p)
    raw_norm_a = math.sqrt(max(raw_norm_a_sq, 0.0))
    raw_norm_p = math.sqrt(max(raw_norm_p_sq, 0.0))

    if fixed_scale and unit_gradient:
        raise ValueError(
            "--mgda-fixed-scale-calibration and --mgda-unit-gradient are mutually exclusive"
        )
    if unit_gradient:
        scale_a = max(raw_norm_a, 1e-12)
        scale_p = max(raw_norm_p, 1e-12)
    elif fixed_scale:
        if calibration_state["acquire_scale"] is None:
            calibration_state["acquire_scale"] = max(raw_norm_a, 1e-12)
            calibration_state["preserve_scale"] = max(raw_norm_p, 1e-12)
        scale_a = calibration_state["acquire_scale"]
        scale_p = calibration_state["preserve_scale"]
    else:
        scale_a = scale_p = 1.0

    calibrated_norm_a_sq = raw_norm_a_sq / (scale_a * scale_a)
    calibrated_norm_p_sq = raw_norm_p_sq / (scale_p * scale_p)
    calibrated_dot = raw_dot / (scale_a * scale_p)
    denominator = max(
        calibrated_norm_a_sq + calibrated_norm_p_sq
        - 2.0 * calibrated_dot,
        1e-30,
    )
    alpha = min(
        1.0,
        max(0.0, (calibrated_norm_p_sq - calibrated_dot) / denominator),
    )

    if unit_gradient:
        # Unit-gradient MGDA determines only the Pareto direction.  Recover the
        # acquire scale so a near-zero preserve gradient cannot shrink the
        # entire slow update to zero.
        common_scale = max(raw_norm_a, 1e-12)
    elif fixed_scale:
        common_scale = math.sqrt(scale_a * scale_p)
    else:
        common_scale = 1.0
    flat_p.mul_((1.0 - alpha) * common_scale / scale_p)
    flat_p.add_(flat_a, alpha=alpha * common_scale / scale_a)
    return {
        "combined": flat_p,
        "alpha": alpha,
        "raw_norm_a_sq": raw_norm_a_sq,
        "raw_norm_p_sq": raw_norm_p_sq,
        "raw_dot": raw_dot,
        "calibrated_norm_a": math.sqrt(max(calibrated_norm_a_sq, 0.0)),
        "calibrated_norm_p": math.sqrt(max(calibrated_norm_p_sq, 0.0)),
        "calibrated_cosine": (
            0.0 if calibrated_norm_a_sq <= 0 or calibrated_norm_p_sq <= 0
            else calibrated_dot
            / math.sqrt(calibrated_norm_a_sq * calibrated_norm_p_sq)
        ),
        "scale_a": scale_a,
        "scale_p": scale_p,
        "common_scale": common_scale,
    }


def four_preference_mgda(flat_gradients, iterations=128, tolerance=1e-10):
    """Min-norm convex combination of four distributed preference gradients.

    The expensive model gradients have already been globally averaged.  The
    remaining optimization is a four-variable simplex QP, solved with
    Frank--Wolfe on its FP64 Gram matrix.  No loss/gradient renormalization is
    applied here: STCH has already placed both task objectives in its detached
    normalized coordinate system.
    """
    count = len(flat_gradients)
    if count != PREFERENCE_COUNT:
        raise ValueError(f"expected {PREFERENCE_COUNT} gradients, got {count}")
    gram = torch.empty((count, count), dtype=torch.float64)
    for i in range(count):
        for j in range(i, count):
            value = flat_norm_dot(flat_gradients[i], flat_gradients[j])
            gram[i, j] = gram[j, i] = value
    alpha = torch.full((count,), 1.0 / count, dtype=torch.float64)
    for _ in range(iterations):
        gram_alpha = gram.mv(alpha)
        vertex = int(torch.argmin(gram_alpha))
        direction = -alpha
        direction[vertex] += 1.0
        denominator = float(direction @ gram.mv(direction))
        if denominator <= 1e-30:
            break
        step = min(1.0, max(0.0, -float(direction @ gram_alpha) / denominator))
        if step <= tolerance:
            break
        alpha.add_(direction, alpha=step)
    combined = torch.zeros_like(flat_gradients[0])
    for weight, gradient in zip(alpha.tolist(), flat_gradients):
        combined.add_(gradient, alpha=weight)
    norms = gram.diag().clamp_min(0).sqrt()
    cosine = torch.zeros_like(gram)
    for i in range(count):
        for j in range(count):
            denominator = float(norms[i] * norms[j])
            cosine[i, j] = 0.0 if denominator == 0.0 else gram[i, j] / denominator
    return {
        "combined": combined,
        "weights": alpha.tolist(),
        "gram": gram.tolist(),
        "norms": norms.tolist(),
        "cosine": cosine.tolist(),
        "common_norm": math.sqrt(max(flat_norm_dot(combined), 0.0)),
    }


def four_preference_common_projection(flat_gradients, alpha=1.0):
    """Remove preference-varying directions from the mean slow gradient.

    Let ``g_bar`` be the mean gradient and ``D=[g_k-g_bar]``.  We return

        g_common = g_bar - alpha D (D^T D)^+ D^T g_bar.

    Since the columns of D sum to zero, ``D^T D`` is singular by
    construction.  ``torch.linalg.pinv`` is therefore intentional here; no
    ``epsilon I`` regularizer is used.  The projection is assembled as a
    linear combination of the already-materialized gradients so it does not
    allocate another K model-sized difference vectors.
    """
    count = len(flat_gradients)
    if count != PREFERENCE_COUNT:
        raise ValueError(f"expected {PREFERENCE_COUNT} gradients, got {count}")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"projection alpha must be in [0, 1], got {alpha}")
    gram = torch.empty((count, count), dtype=torch.float64)
    for i in range(count):
        for j in range(i, count):
            value = flat_norm_dot(flat_gradients[i], flat_gradients[j])
            gram[i, j] = gram[j, i] = value

    identity = torch.eye(count, dtype=torch.float64)
    ones = torch.ones((count, count), dtype=torch.float64) / count
    center = identity - ones
    mean_coefficients = torch.full((count,), 1.0 / count, dtype=torch.float64)
    difference_gram = center @ gram @ center
    difference_rhs = center @ gram @ mean_coefficients
    coefficients = torch.linalg.pinv(
        difference_gram, hermitian=True,
    ) @ difference_rhs
    hard_weights = (
        (1.0 + coefficients.sum()) / count - coefficients
    )
    # alpha=0 recovers the ordinary mean gradient; alpha=1 is the exact
    # orthogonal projection.  Intermediate values retain part of the shared
    # learning signal when the preference-difference span nearly contains the
    # mean gradient.
    weights = (
        (1.0 - float(alpha)) * mean_coefficients
        + float(alpha) * hard_weights
    )

    combined = torch.zeros_like(flat_gradients[0])
    mean_gradient = torch.zeros_like(flat_gradients[0])
    for weight, gradient in zip(weights.tolist(), flat_gradients):
        combined.add_(gradient, alpha=weight)
        mean_gradient.add_(gradient, alpha=1.0 / count)

    common_norm_sq = flat_norm_dot(combined)
    mean_norm_sq = flat_norm_dot(mean_gradient)
    removed = mean_gradient.sub(combined)
    removed_norm_sq = flat_norm_dot(removed)
    # Every preference should have the same first-order directional
    # derivative along the projected common update.
    directional = torch.tensor([
        flat_norm_dot(gradient, combined) for gradient in flat_gradients
    ], dtype=torch.float64)
    eigenvalues = torch.linalg.eigvalsh(difference_gram)
    tolerance = (
        max(difference_gram.shape)
        * torch.finfo(difference_gram.dtype).eps
        * eigenvalues.abs().max().clamp_min(1.0)
    )
    rank = int((eigenvalues > tolerance).sum())
    norms = gram.diag().clamp_min(0).sqrt()
    cosine = torch.zeros_like(gram)
    for i in range(count):
        for j in range(count):
            denominator = float(norms[i] * norms[j])
            cosine[i, j] = (
                0.0 if denominator == 0.0 else gram[i, j] / denominator
            )
    return {
        "combined": combined,
        "weights": weights.tolist(),
        "gram": gram.tolist(),
        "difference_gram": difference_gram.tolist(),
        "difference_rank": rank,
        "difference_eigenvalues": eigenvalues.tolist(),
        "norms": norms.tolist(),
        "cosine": cosine.tolist(),
        "common_norm": math.sqrt(max(common_norm_sq, 0.0)),
        "mean_norm": math.sqrt(max(mean_norm_sq, 0.0)),
        "removed_norm": math.sqrt(max(removed_norm_sq, 0.0)),
        "removed_ratio": (
            0.0 if mean_norm_sq <= 0 else
            math.sqrt(max(removed_norm_sq, 0.0) / mean_norm_sq)
        ),
        "directional_derivatives": directional.tolist(),
        "directional_spread": float(directional.max() - directional.min()),
        "projection_alpha": float(alpha),
        "hard_projection_removed_ratio": (
            0.0 if float(alpha) == 0.0 else
            (0.0 if mean_norm_sq <= 0 else
             math.sqrt(max(removed_norm_sq, 0.0) / mean_norm_sq) / float(alpha))
        ),
    }


def dwa_direction(flat_a, flat_p, calibration_state, dwa_state,
                  acquire_loss, preserve_loss, temperature, window):
    """DWA sum of endpoint gradients normalized by first-batch scales."""
    raw_norm_a_sq = flat_norm_dot(flat_a)
    raw_norm_p_sq = flat_norm_dot(flat_p)
    raw_dot = flat_norm_dot(flat_a, flat_p)
    raw_norm_a = math.sqrt(max(raw_norm_a_sq, 0.0))
    raw_norm_p = math.sqrt(max(raw_norm_p_sq, 0.0))
    if calibration_state["acquire_scale"] is None:
        calibration_state["acquire_scale"] = max(raw_norm_a, 1e-12)
        calibration_state["preserve_scale"] = max(raw_norm_p, 1e-12)
    scale_a = calibration_state["acquire_scale"]
    scale_p = calibration_state["preserve_scale"]

    history_a = dwa_state["acquire_losses"]
    history_p = dwa_state["preserve_losses"]
    ratio_a = ratio_p = 1.0
    if len(history_a) >= 2 * window:
        older_a = sum(history_a[-2 * window:-window]) / window
        newer_a = sum(history_a[-window:]) / window
        older_p = sum(history_p[-2 * window:-window]) / window
        newer_p = sum(history_p[-window:]) / window
        ratio_a = newer_a / max(older_a, 1e-12)
        ratio_p = newer_p / max(older_p, 1e-12)
    logits = torch.tensor(
        [ratio_a / temperature, ratio_p / temperature], dtype=torch.float64,
    )
    weight_a, weight_p = (2.0 * logits.softmax(dim=0)).tolist()
    history_a.append(float(acquire_loss))
    history_p.append(float(preserve_loss))

    common_scale = math.sqrt(scale_a * scale_p)
    # DWA weights sum to two. Divide by two so the update magnitude remains
    # comparable to a convex MGDA combination.
    flat_p.mul_(0.5 * weight_p * common_scale / scale_p)
    flat_p.add_(flat_a, alpha=0.5 * weight_a * common_scale / scale_a)
    return {
        "combined": flat_p,
        "alpha": None,
        "raw_norm_a_sq": raw_norm_a_sq,
        "raw_norm_p_sq": raw_norm_p_sq,
        "raw_dot": raw_dot,
        "calibrated_norm_a": raw_norm_a / scale_a,
        "calibrated_norm_p": raw_norm_p / scale_p,
        "scale_a": scale_a,
        "scale_p": scale_p,
        "common_scale": common_scale,
        "dwa_weight_acquire": float(weight_a),
        "dwa_weight_preserve": float(weight_p),
        "dwa_ratio_acquire": ratio_a,
        "dwa_ratio_preserve": ratio_p,
    }


def acquire_primary_projection(flat_a, flat_p, eps=1e-30):
    """Project acquire gradient onto the preserve-safe half-space.

    With theta <- theta - eta*d, preserve is non-increasing to first order
    whenever g_p^T d >= 0.  The closest feasible direction to g_a has the
    closed form below.
    """
    norm_a_sq = flat_norm_dot(flat_a)
    norm_p_sq = flat_norm_dot(flat_p)
    dot_before = flat_norm_dot(flat_a, flat_p)
    active = dot_before < 0.0 and norm_p_sq > eps
    combined = flat_a.clone()
    if active:
        combined.add_(flat_p, alpha=-dot_before / (norm_p_sq + eps))
    dot_after = flat_norm_dot(combined, flat_p)
    correction_sq = flat_norm_dot(combined - flat_a)
    return {
        "combined": combined,
        "raw_norm_a_sq": norm_a_sq,
        "raw_norm_p_sq": norm_p_sq,
        "raw_dot": dot_before,
        "dot_after": dot_after,
        "active": active,
        "removed_ratio": math.sqrt(max(correction_sq, 0.0))
        / max(math.sqrt(max(norm_a_sq, 0.0)), 1e-30),
    }


def allreduce_small_grads(parameters):
    world = dist.get_world_size()
    for parameter in parameters:
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world)


def joint_random_preference_update(
    model, code_model, tensors, completions, completion_mask,
    slow_optimizer, fast_optimizer, slow_scheduler, fast_scheduler,
    args, mgda_calibration_state, dwa_state,
):
    """Jointly update theta/phi at one batch-shared random preference.

    Acquire and preserve gradients are formed over the same conditional model
    pi_{theta,z_phi(lambda)}.  Conflict geometry acts on their concatenated
    (theta, phi) gradient vectors, so both optimizers follow exactly the same
    DWA/projection decision.  No STCH scalarization is used in this phase.
    """
    model.requires_grad_(True)
    code_model.requires_grad_(True)
    model.train()
    slow_optimizer.zero_grad(set_to_none=True)
    fast_optimizer.zero_grad(set_to_none=True)
    slow_parameters = [p for p in model.parameters() if p.requires_grad]
    fast_parameters = [p for p in code_model.parameters() if p.requires_grad]
    parameters = slow_parameters + fast_parameters
    acquire_grads = [None] * len(parameters)
    preserve_grads = [None] * len(parameters)
    count = len(tensors["lambdas"])
    acquire_sum = preserve_sum = 0.0

    for start in range(0, count, args.condition_chunk):
        end = min(start + args.condition_chunk, count)
        sliced = slice_tensors(tensors, start, end)
        acquire, preserve = objective_chunks(
            model, None, code_model, sliced, completions[start:end],
            completion_mask[start:end], args, need_preserve=True,
        )
        grad_a = torch.autograd.grad(
            acquire.sum() / count, parameters, retain_graph=True,
            allow_unused=True,
        )
        grad_p = torch.autograd.grad(
            preserve.sum() / count, parameters, retain_graph=False,
            allow_unused=True,
        )
        add_gradients_(parameters, grad_a, acquire_grads)
        add_gradients_(parameters, grad_p, preserve_grads)
        acquire_sum += float(acquire.detach().sum())
        preserve_sum += float(preserve.detach().sum())
        del sliced, acquire, preserve, grad_a, grad_p

    split = len(slow_parameters)
    for parameter, gradient in zip(parameters, acquire_grads):
        parameter.grad = torch.zeros_like(parameter) if gradient is None else gradient
    flat_a_slow = fused_allreduce_parameter_grads(
        slow_parameters, dist.get_world_size(),
    ).detach().clone()
    flat_a_fast = fused_allreduce_parameter_grads(
        fast_parameters, dist.get_world_size(),
    ).detach().clone()
    # Geometry needs one vector, but model gradients are bf16 while prompt
    # controls are deliberately fp32.  Use the model dtype only for the small
    # geometry vector concatenation, then restore each parameter's native
    # dtype when assigning the combined direction.
    flat_a = torch.cat((
        flat_a_slow, flat_a_fast.to(flat_a_slow.dtype),
    ))
    for parameter, gradient in zip(parameters, preserve_grads):
        parameter.grad = torch.zeros_like(parameter) if gradient is None else gradient
    flat_p_slow = fused_allreduce_parameter_grads(
        slow_parameters, dist.get_world_size(),
    )
    flat_p_fast = fused_allreduce_parameter_grads(
        fast_parameters, dist.get_world_size(),
    )
    flat_p = torch.cat((
        flat_p_slow, flat_p_fast.to(flat_p_slow.dtype),
    ))

    loss_a, loss_p = acquire_sum / count, preserve_sum / count
    if args.slow_dwa:
        result = dwa_direction(
            flat_a, flat_p, mgda_calibration_state, dwa_state,
            loss_a, loss_p, args.dwa_temperature, args.dwa_window,
        )
        geometry, projection, alpha = result, None, None
    elif args.slow_gradient_projection:
        result = acquire_primary_projection(flat_a, flat_p)
        geometry, projection, alpha = None, result, None
    else:
        raise ValueError(
            "joint_random_then_fast requires --slow-dwa or "
            "--slow-gradient-projection"
        )
    slow_elements = flat_a_slow.numel()
    combined_slow = torch._utils._unflatten_dense_tensors(
        result["combined"][:slow_elements],
        [p.grad for p in slow_parameters],
    )
    combined_fast = torch._utils._unflatten_dense_tensors(
        result["combined"][slow_elements:].to(flat_a_fast.dtype),
        [p.grad for p in fast_parameters],
    )
    for parameter, gradient in zip(slow_parameters, combined_slow):
        parameter.grad = gradient
    for parameter, gradient in zip(fast_parameters, combined_fast):
        parameter.grad = gradient
    joint_grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters, args.max_grad_norm,
    )
    slow_optimizer.step()
    fast_optimizer.step()
    slow_scheduler.step()
    fast_scheduler.step()
    slow_optimizer.zero_grad(set_to_none=True)
    fast_optimizer.zero_grad(set_to_none=True)

    norm_a = result["raw_norm_a_sq"]
    norm_p = result["raw_norm_p_sq"]
    dot = result["raw_dot"]
    return {
        "joint_random_lambda": float(tensors["lambdas"][0]),
        "joint_acquire_sft_nll": loss_a,
        "joint_preserve_sft_nll": loss_p,
        "joint_gradient_cosine": (
            0.0 if norm_a <= 0 or norm_p <= 0
            else dot / math.sqrt(norm_a * norm_p)
        ),
        "joint_grad_norm_acquire": math.sqrt(max(norm_a, 0.0)),
        "joint_grad_norm_preserve": math.sqrt(max(norm_p, 0.0)),
        "joint_grad_norm_preclip": float(joint_grad_norm),
        "projection_active": None if projection is None else projection["active"],
        "projection_removed_ratio": (
            None if projection is None else projection["removed_ratio"]
        ),
        "dwa_weight_acquire": (
            None if geometry is None else geometry.get("dwa_weight_acquire")
        ),
        "dwa_weight_preserve": (
            None if geometry is None else geometry.get("dwa_weight_preserve")
        ),
        "dwa_ratio_acquire": (
            None if geometry is None else geometry.get("dwa_ratio_acquire")
        ),
        "dwa_ratio_preserve": (
            None if geometry is None else geometry.get("dwa_ratio_preserve")
        ),
        "joint_slow_lr": slow_optimizer.param_groups[0]["lr"],
        "joint_fast_lr": fast_optimizer.param_groups[0]["lr"],
    }


def fixed_uniform_slow_update(
    model, code_model, tensors, optimizer, scheduler, args,
    preserve_teacher=None, transport_source_prompt=None,
):
    """Train only slow weights at the fixed, meaningful midpoint lambda=0.5.

    The fast path is present in the forward graph but frozen, so the slow
    model is optimized in exactly the coordinate system used by Phase B and
    evaluation.  Objective scales are fixed in nats/token for the whole stage;
    no moving batch min/range can rotate the preference geometry.
    """
    if args.fixed_acquire_scale is None or args.fixed_preserve_scale is None:
        raise ValueError(
            "uniform_then_fast requires --fixed-acquire-scale and "
            "--fixed-preserve-scale"
        )
    if args.fixed_acquire_scale <= 0 or args.fixed_preserve_scale <= 0:
        raise ValueError("fixed functional scales must be positive")
    model.requires_grad_(True)
    code_model.requires_grad_(False)
    model.train()
    code_model.eval()
    optimizer.zero_grad(set_to_none=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    count = len(tensors["lambdas"])
    loss_sum = acquire_sum = preserve_sum = 0.0
    effective_a_sum = 0.0
    entropy_sum = 0.0
    ideals = torch.zeros(2, device=tensors["lambdas"].device)
    ranges = torch.tensor(
        [args.fixed_acquire_scale, args.fixed_preserve_scale],
        device=tensors["lambdas"].device,
    )
    for start in range(0, count, args.condition_chunk):
        end = min(start + args.condition_chunk, count)
        sliced = slice_tensors(tensors, start, end)
        acquire, preserve = objective_chunks(
            model, None, code_model, sliced,
            tensors["current_target_ids"][start:end],
            tensors["current_target_mask"][start:end],
            args, need_preserve=True, preserve_teacher=preserve_teacher,
            transport_source_prompt=transport_source_prompt,
        )
        values, weights, entropy = stch_loss(
            acquire, preserve, tensors["lambdas"][start:end],
            ideals, ranges, args.stch_mu,
        )
        (values.sum() / count).backward()
        loss_sum += float(values.detach().sum())
        acquire_sum += float(acquire.detach().sum())
        preserve_sum += float(preserve.detach().sum())
        effective_a_sum += float(weights[:, 0].detach().sum())
        entropy_sum += float(entropy.detach().sum())
        del sliced, acquire, preserve, values, weights, entropy
    fused_allreduce_parameter_grads(parameters, dist.get_world_size())
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "slow_uniform_lambda": 0.5,
        "slow_uniform_loss": loss_sum / count,
        "slow_acquire_sft_nll": acquire_sum / count,
        "slow_preserve_top64_other_fkl": preserve_sum / count,
        "slow_effective_acquire_weight": effective_a_sum / count,
        "slow_stch_entropy": entropy_sum / count,
        "fixed_functional_scales": ranges.detach().cpu().tolist(),
        "constant_preserve_target_cache": bool(
            tensors.get("constant_preserve_target_cache", False)
        ),
        "slow_grad_norm_preclip": float(grad_norm),
        "slow_lr": optimizer.param_groups[0]["lr"],
    }


def fast_update(model, teacher, code_model, tensors, completions,
                completion_mask, optimizer, scheduler, args, stage,
                normalization_state, preserve_teacher=None,
                transport_source_prompt=None):
    model.requires_grad_(False)
    code_model.requires_grad_(True)
    if hasattr(code_model, "text_encoder"):
        code_model.text_encoder.requires_grad_(False)
        code_model.text_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    lambdas = tensors["lambdas"]
    count = len(lambdas)
    # Normalize step t with statistics measured at step t-1.  This removes the
    # old graph-free duplicate forward while keeping normalization detached.
    if stage == 0:
        minima = maxima = ranges = torch.zeros(2, device=lambdas.device)
    elif (args.fixed_acquire_scale is not None
          or args.fixed_preserve_scale is not None):
        if args.fixed_acquire_scale is None or args.fixed_preserve_scale is None:
            raise ValueError("both fixed objective scales must be specified")
        minima = torch.zeros(2, device=lambdas.device)
        ranges = torch.tensor(
            [args.fixed_acquire_scale, args.fixed_preserve_scale],
            device=lambdas.device,
        )
        maxima = ranges
    else:
        minima = normalization_state["minima"].to(lambdas.device)
        ranges = normalization_state["ranges"].to(lambdas.device)
        maxima = minima + ranges
    observed_min = torch.full((2,), float("inf"), device=lambdas.device)
    observed_max = torch.full((2,), float("-inf"), device=lambdas.device)

    loss_sum = acquire_sum = preserve_sum = weight_sum = entropy_sum = 0.0
    for start in range(0, count, args.condition_chunk):
        end = min(start + args.condition_chunk, count)
        sliced = slice_tensors(tensors, start, end)
        acquire, preserve = objective_chunks(
            model, teacher, code_model, sliced, completions[start:end],
            completion_mask[start:end], args, need_preserve=stage > 0,
            preserve_teacher=preserve_teacher,
            transport_source_prompt=transport_source_prompt,
        )
        if stage == 0:
            values = acquire
            weights = torch.ones_like(acquire)
            entropy = torch.zeros_like(acquire)
        elif args.fast_scalarization == "linear":
            chunk_lambdas = lambdas[start:end]
            values = (1.0 - chunk_lambdas) * acquire + chunk_lambdas * preserve
            weights = 1.0 - chunk_lambdas
            entropy = -(
                weights * weights.clamp_min(1e-12).log()
                + chunk_lambdas * chunk_lambdas.clamp_min(1e-12).log()
            )
            detached = torch.stack((acquire.detach(), preserve.detach()))
            observed_min = torch.minimum(observed_min, detached.amin(dim=1))
            observed_max = torch.maximum(observed_max, detached.amax(dim=1))
        else:
            stch_minima = minima if args.stch_normalization else torch.zeros_like(minima)
            stch_ranges = ranges if args.stch_normalization else torch.ones_like(ranges)
            values, stch_weights, entropy = stch_loss(
                acquire, preserve, lambdas[start:end],
                0.95 * stch_minima, stch_ranges, args.stch_mu,
            )
            weights = stch_weights[:, 0]
            detached = torch.stack((acquire.detach(), preserve.detach()))
            observed_min = torch.minimum(observed_min, detached.amin(dim=1))
            observed_max = torch.maximum(observed_max, detached.amax(dim=1))
        (values.sum() / count).backward()
        loss_sum += float(values.detach().sum())
        acquire_sum += float(acquire.detach().sum())
        preserve_sum += 0.0 if preserve is None else float(preserve.detach().sum())
        weight_sum += float(weights.detach().sum())
        entropy_sum += float(entropy.detach().sum())
        del sliced, acquire, preserve, values, weights, entropy
    parameters = [p for p in code_model.parameters() if p.requires_grad]
    allreduce_small_grads(parameters)
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    if (stage > 0 and args.stch_normalization
            and args.fixed_acquire_scale is None
            and args.fast_scalarization == "stch"):
        dist.all_reduce(observed_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(observed_max, op=dist.ReduceOp.MAX)
        normalization_state["minima"] = observed_min.detach().cpu()
        normalization_state["ranges"] = (
            observed_max - observed_min
        ).clamp_min(args.range_floor).detach().cpu()
    return {
        "fast_loss": loss_sum / count,
        "acquire_fkl": acquire_sum / count,
        "preserve_sft_nll": (
            preserve_sum / count
            if stage > 0 and args.preserve_objective == "sft" else None
        ),
        "preserve_top64_other_fkl": (
            preserve_sum / count
            if stage > 0 and args.preserve_objective == "topk_fkl" else None
        ),
        "fast_effective_acquire_weight": weight_sum / count,
        "stch_entropy": entropy_sum / count,
        "stch_min": [float(x) for x in minima],
        "stch_max": [float(x) for x in maxima],
        "stch_range": [float(x) for x in ranges],
        "normalization_delay_steps": (
            1 if (args.stch_normalization
                  and args.fixed_acquire_scale is None
                  and args.fast_scalarization == "stch") else 0
        ),
        "fast_grad_norm": float(grad_norm),
        "fast_lr": optimizer.param_groups[0]["lr"],
        "fast_frozen": False,
        "constant_preserve_target_cache": bool(
            tensors.get("constant_preserve_target_cache", False)
        ),
    }


def backward_acquire_endpoint(model, teacher, code_model, tensors, completions,
                              completion_mask, args):
    indices = torch.arange(
        0, len(tensors["lambdas"]), PREFERENCE_COUNT,
        device=completions.device,
    )
    losses = []
    for start in range(0, len(indices), args.condition_chunk):
        chosen = indices[start:start + args.condition_chunk]
        code = code_model(torch.zeros(len(chosen), device=completions.device)).detach()
        if args.acquire_objective == "sft":
            student_logits = completion_logits(
                model, tensors["prompt_ids"][chosen], tensors["prompt_mask"][chosen],
                tensors["current_target_ids"][chosen],
                tensors["current_target_mask"][chosen], code,
                tensors["positions"][chosen],
            )
            value = sft_nll_per_example(
                student_logits, tensors["current_target_ids"][chosen],
                tensors["current_target_mask"][chosen],
            )
        else:
            teacher_logits = teacher_completion_logits(
                teacher, tensors["teacher_ids"][chosen], tensors["teacher_mask"][chosen],
                completions[chosen], completion_mask[chosen],
            )
            student_logits = completion_logits(
                model, tensors["prompt_ids"][chosen], tensors["prompt_mask"][chosen],
                completions[chosen], completion_mask[chosen], code,
                tensors["positions"][chosen],
            )
            value = acquire_kl_per_example(
                teacher_logits, student_logits, completion_mask[chosen], args,
            )
        (value.sum() / len(indices)).backward()
        losses.append(value.detach())
        del student_logits
        if args.acquire_objective != "sft":
            del teacher_logits
    return torch.cat(losses).mean()


def backward_preserve_endpoint(model, code_model, tensors, args):
    indices = torch.arange(
        PREFERENCE_COUNT - 1, len(tensors["lambdas"]), PREFERENCE_COUNT,
        device=tensors["lambdas"].device,
    )
    losses = []
    for start in range(0, len(indices), args.condition_chunk):
        chosen = indices[start:start + args.condition_chunk]
        code = code_model(torch.ones(len(chosen), device=indices.device)).detach()
        logits = completion_logits(
            model, tensors["replay_ids"][chosen], tensors["replay_mask"][chosen],
            tensors["target_ids"][chosen], tensors["target_mask"][chosen],
            code, tensors["replay_positions"][chosen],
        )
        value = sft_nll_per_example(
            logits, tensors["target_ids"][chosen], tensors["target_mask"][chosen],
        )
        (value.sum() / len(indices)).backward()
        losses.append(value.detach())
        del logits
    return torch.cat(losses).mean()


def slow_mgda_update(model, teacher, code_model, tensors, completions,
                     completion_mask, optimizer, scheduler, args, stage,
                     mgda_calibration_state):
    code_model.requires_grad_(False)
    model.requires_grad_(True)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    acquire = backward_acquire_endpoint(
        model, teacher, code_model, tensors, completions, completion_mask, args,
    )
    flat_a = fused_allreduce_parameter_grads(parameters, dist.get_world_size()).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    geometry = projection = None
    if stage == 0:
        grad_a = torch._utils._unflatten_dense_tensors(flat_a, parameters)
        for parameter, gradient in zip(parameters, grad_a):
            parameter.grad = gradient
        alpha = 1.0
        preserve = None
        norm_a = flat_norm_dot(flat_a)
        norm_p = dot = 0.0
    else:
        preserve = backward_preserve_endpoint(model, code_model, tensors, args)
        flat_p = fused_allreduce_parameter_grads(parameters, dist.get_world_size())
        if args.slow_gradient_projection:
            projection = acquire_primary_projection(flat_a, flat_p)
            result = projection
            alpha = None
        else:
            geometry = calibrated_mgda(
                flat_a, flat_p, mgda_calibration_state,
                args.mgda_fixed_scale_calibration, args.mgda_unit_gradient,
            )
            result = geometry
            alpha = geometry["alpha"]
        norm_a = result["raw_norm_a_sq"]
        norm_p = result["raw_norm_p_sq"]
        dot = result["raw_dot"]
        combined = torch._utils._unflatten_dense_tensors(
            result["combined"], parameters,
        )
        for parameter, gradient in zip(parameters, combined):
            parameter.grad = gradient
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "slow_acquire_fkl_lambda0": float(acquire),
        "slow_preserve_sft_nll_lambda1": None if preserve is None else float(preserve),
        "mgda_alpha_acquire": alpha,
        "grad_norm_acquire": math.sqrt(max(norm_a, 0.0)),
        "grad_norm_preserve": math.sqrt(max(norm_p, 0.0)),
        "gradient_cosine": (
            0.0 if norm_a <= 0 or norm_p <= 0
            else dot / math.sqrt(norm_a * norm_p)
        ),
        "mgda_calibration_acquire_scale": (
            None if geometry is None else geometry["scale_a"]
        ),
        "mgda_calibration_preserve_scale": (
            None if geometry is None else geometry["scale_p"]
        ),
        "calibrated_grad_norm_acquire": (
            None if geometry is None else geometry["calibrated_norm_a"]
        ),
        "calibrated_grad_norm_preserve": (
            None if geometry is None else geometry["calibrated_norm_p"]
        ),
        "mgda_common_scale": (
            None if geometry is None else geometry["common_scale"]
        ),
        "projection_active": None if projection is None else projection["active"],
        "projection_removed_ratio": (
            None if projection is None else projection["removed_ratio"]
        ),
        "preserve_dot_after_projection": (
            None if projection is None else projection["dot_after"]
        ),
        "slow_grad_norm_preclip": float(grad_norm),
        "slow_lr": scheduler.get_last_lr()[0],
    }


def add_gradients_(parameters, gradients, accumulator=None):
    if accumulator is None:
        for parameter, gradient in zip(parameters, gradients):
            if gradient is not None:
                if parameter.grad is None:
                    parameter.grad = gradient.detach()
                else:
                    parameter.grad.add_(gradient.detach())
        return None
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        if accumulator[index] is None:
            accumulator[index] = gradient.detach()
        else:
            accumulator[index].add_(gradient.detach())
    return accumulator


def joint_reused_update(model, teacher, code_model, tensors, completions,
                        completion_mask, fast_optimizer, slow_optimizer,
                        scheduler, args, stage, normalization_state,
                        mgda_calibration_state):
    """Fast STCH and slow endpoint MGDA from the same forward graphs.

    The old implementation recomputed lambda=0/1 forwards after the STCH
    update. Here each endpoint graph supplies both its fast-memory gradient and
    its slow-memory MGDA gradient before either optimizer steps.
    """
    model.requires_grad_(True)
    code_model.requires_grad_(True)
    model.train()
    fast_optimizer.zero_grad(set_to_none=True)
    slow_optimizer.zero_grad(set_to_none=True)
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    code_parameters = [p for p in code_model.parameters() if p.requires_grad]
    preserve_gradients = [None] * len(model_parameters) if stage > 0 else None
    lambdas = tensors["lambdas"]
    count = len(lambdas)
    question_count = count // PREFERENCE_COUNT

    if stage == 0:
        minima = maxima = ranges = torch.zeros(2, device=lambdas.device)
    else:
        minima = normalization_state["minima"].to(lambdas.device)
        ranges = normalization_state["ranges"].to(lambdas.device)
        maxima = minima + ranges
    observed_min = torch.full((2,), float("inf"), device=lambdas.device)
    observed_max = torch.full((2,), float("-inf"), device=lambdas.device)
    loss_sum = acquire_sum = preserve_sum = weight_sum = entropy_sum = 0.0
    endpoint_acquire_sum = endpoint_preserve_sum = 0.0

    for start in range(0, count, args.condition_chunk):
        end = min(start + args.condition_chunk, count)
        sliced = slice_tensors(tensors, start, end)
        acquire, preserve = objective_chunks(
            model, teacher, code_model, sliced, completions[start:end],
            completion_mask[start:end], args, need_preserve=stage > 0,
        )
        if stage == 0:
            values = acquire
            weights = torch.ones_like(acquire)
            entropy = torch.zeros_like(acquire)
        else:
            values, stch_weights, entropy = stch_loss(
                acquire, preserve, lambdas[start:end],
                0.95 * minima, ranges, args.stch_mu,
            )
            weights = stch_weights[:, 0]
            detached = torch.stack((acquire.detach(), preserve.detach()))
            observed_min = torch.minimum(observed_min, detached.amin(dim=1))
            observed_max = torch.maximum(observed_max, detached.amax(dim=1))

        code_gradients = torch.autograd.grad(
            values.sum() / count, code_parameters, retain_graph=True,
            allow_unused=True,
        )
        add_gradients_(code_parameters, code_gradients)

        global_indices = torch.arange(start, end, device=lambdas.device)
        acquire_positions = (
            global_indices.remainder(PREFERENCE_COUNT) == 0
        ).nonzero().flatten()
        preserve_positions = (
            global_indices.remainder(PREFERENCE_COUNT) == PREFERENCE_COUNT - 1
        ).nonzero().flatten()
        if acquire_positions.numel():
            endpoint_a = acquire[acquire_positions].sum() / question_count
            gradients_a = torch.autograd.grad(
                endpoint_a, model_parameters,
                retain_graph=bool(stage > 0 and preserve_positions.numel()),
                allow_unused=True,
            )
            add_gradients_(model_parameters, gradients_a)
            endpoint_acquire_sum += float(acquire[acquire_positions].detach().sum())
        if stage > 0 and preserve_positions.numel():
            endpoint_p = preserve[preserve_positions].sum() / question_count
            gradients_p = torch.autograd.grad(
                endpoint_p, model_parameters, retain_graph=False,
                allow_unused=True,
            )
            add_gradients_(model_parameters, gradients_p, preserve_gradients)
            endpoint_preserve_sum += float(preserve[preserve_positions].detach().sum())

        loss_sum += float(values.detach().sum())
        acquire_sum += float(acquire.detach().sum())
        preserve_sum += 0.0 if preserve is None else float(preserve.detach().sum())
        weight_sum += float(weights.detach().sum())
        entropy_sum += float(entropy.detach().sum())
        del sliced, acquire, preserve, values, weights, entropy

    allreduce_small_grads(code_parameters)
    fast_grad_norm = torch.nn.utils.clip_grad_norm_(
        code_parameters, args.max_grad_norm,
    )

    flat_a = fused_allreduce_parameter_grads(
        model_parameters, dist.get_world_size(),
    ).detach().clone()
    geometry = projection = None
    if stage == 0:
        norm_a = flat_norm_dot(flat_a)
        norm_p = dot = 0.0
        alpha = 1.0
    else:
        for parameter, gradient in zip(model_parameters, preserve_gradients):
            parameter.grad = (
                torch.zeros_like(parameter) if gradient is None else gradient
            )
        flat_p = fused_allreduce_parameter_grads(
            model_parameters, dist.get_world_size(),
        )
        if args.slow_gradient_projection:
            projection = acquire_primary_projection(flat_a, flat_p)
            result = projection
            alpha = None
        else:
            geometry = calibrated_mgda(
                flat_a, flat_p, mgda_calibration_state,
                args.mgda_fixed_scale_calibration, args.mgda_unit_gradient,
            )
            result = geometry
            alpha = geometry["alpha"]
        norm_a = result["raw_norm_a_sq"]
        norm_p = result["raw_norm_p_sq"]
        dot = result["raw_dot"]
        combined = torch._utils._unflatten_dense_tensors(
            result["combined"], model_parameters,
        )
        for parameter, gradient in zip(model_parameters, combined):
            parameter.grad = gradient

    slow_grad_norm = torch.nn.utils.clip_grad_norm_(
        model_parameters, args.max_grad_norm,
    )
    fast_optimizer.step()
    slow_optimizer.step()
    scheduler.step()
    fast_optimizer.zero_grad(set_to_none=True)
    slow_optimizer.zero_grad(set_to_none=True)

    if stage > 0:
        dist.all_reduce(observed_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(observed_max, op=dist.ReduceOp.MAX)
        normalization_state["minima"] = observed_min.detach().cpu()
        normalization_state["ranges"] = (
            observed_max - observed_min
        ).clamp_min(args.range_floor).detach().cpu()

    return {
        "fast_loss": loss_sum / count,
        "acquire_fkl": acquire_sum / count,
        "preserve_sft_nll": None if stage == 0 else preserve_sum / count,
        "fast_effective_acquire_weight": weight_sum / count,
        "stch_entropy": entropy_sum / count,
        "stch_min": [float(x) for x in minima],
        "stch_max": [float(x) for x in maxima],
        "stch_range": [float(x) for x in ranges],
        "normalization_delay_steps": 1,
        "fast_grad_norm": float(fast_grad_norm),
        "slow_acquire_fkl_lambda0": endpoint_acquire_sum / question_count,
        "slow_preserve_sft_nll_lambda1": (
            None if stage == 0 else endpoint_preserve_sum / question_count
        ),
        "mgda_alpha_acquire": alpha,
        "grad_norm_acquire": math.sqrt(max(norm_a, 0.0)),
        "grad_norm_preserve": math.sqrt(max(norm_p, 0.0)),
        "gradient_cosine": (
            0.0 if norm_a <= 0 or norm_p <= 0
            else dot / math.sqrt(norm_a * norm_p)
        ),
        "mgda_calibration_acquire_scale": (
            None if geometry is None else geometry["scale_a"]
        ),
        "mgda_calibration_preserve_scale": (
            None if geometry is None else geometry["scale_p"]
        ),
        "calibrated_grad_norm_acquire": (
            None if geometry is None else geometry["calibrated_norm_a"]
        ),
        "calibrated_grad_norm_preserve": (
            None if geometry is None else geometry["calibrated_norm_p"]
        ),
        "mgda_common_scale": (
            None if geometry is None else geometry["common_scale"]
        ),
        "projection_active": None if projection is None else projection["active"],
        "projection_removed_ratio": (
            None if projection is None else projection["removed_ratio"]
        ),
        "preserve_dot_after_projection": (
            None if projection is None else projection["dot_after"]
        ),
        "slow_grad_norm_preclip": float(slow_grad_norm),
        "slow_lr": scheduler.get_last_lr()[0],
        "endpoint_forward_reused": True,
    }


def joint_four_preference_mgda_stch_update(
    model, teacher, code_model, tensors, completions, completion_mask,
    fast_optimizer, slow_optimizer, slow_scheduler, fast_scheduler,
    args, stage, normalization_state,
):
    """One shared-forward joint step: slow 4-pref MGDA, fast mean STCH.

    For each batch-shared preference lambda_j we form the scalarized STCH
    objective l_j.  The backbone follows the min-norm convex combination of
    {grad_theta l_j}; the prompt follows grad_phi mean_j(l_j).  All gradients
    are extracted before either optimizer steps, so both blocks see the same
    parameter point and reuse exactly the same forward graphs.
    """
    model.requires_grad_(True)
    code_model.requires_grad_(True)
    model.train()
    fast_optimizer.zero_grad(set_to_none=True)
    slow_optimizer.zero_grad(set_to_none=True)
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    code_parameters = [p for p in code_model.parameters() if p.requires_grad]
    gradient_count = 2 if args.slow_endpoint_safe_projection else PREFERENCE_COUNT
    preference_gradients = [
        [None] * len(model_parameters) for _ in range(gradient_count)
    ]
    lambdas = tensors["lambdas"]
    count = len(lambdas)
    question_count = count // PREFERENCE_COUNT

    if stage == 0:
        minima = maxima = ranges = torch.zeros(2, device=lambdas.device)
    else:
        minima = normalization_state["minima"].to(lambdas.device)
        ranges = normalization_state["ranges"].to(lambdas.device)
        maxima = minima + ranges
    observed_min = torch.full((2,), float("inf"), device=lambdas.device)
    observed_max = torch.full((2,), float("-inf"), device=lambdas.device)
    loss_sum = acquire_sum = preserve_sum = weight_sum = entropy_sum = 0.0
    endpoint_acquire_sum = endpoint_preserve_sum = 0.0

    for start in range(0, count, args.condition_chunk):
        end = min(start + args.condition_chunk, count)
        sliced = slice_tensors(tensors, start, end)
        acquire, preserve = objective_chunks(
            model, teacher, code_model, sliced, completions[start:end],
            completion_mask[start:end], args, need_preserve=stage > 0,
        )
        if stage == 0:
            values = acquire
            weights = torch.ones_like(acquire)
            entropy = torch.zeros_like(acquire)
        else:
            values, stch_weights, entropy = stch_loss(
                acquire, preserve, lambdas[start:end],
                0.95 * minima, ranges, args.stch_mu,
            )
            weights = stch_weights[:, 0]
            detached = torch.stack((acquire.detach(), preserve.detach()))
            observed_min = torch.minimum(observed_min, detached.amin(dim=1))
            observed_max = torch.maximum(observed_max, detached.amax(dim=1))

        code_gradients = torch.autograd.grad(
            values.sum() / count, code_parameters, retain_graph=True,
            allow_unused=True,
        )
        add_gradients_(code_parameters, code_gradients)

        global_indices = torch.arange(start, end, device=lambdas.device)
        if args.slow_endpoint_safe_projection:
            endpoint_objectives = []
            acquire_positions = (
                global_indices.remainder(PREFERENCE_COUNT) == 0
            ).nonzero().flatten()
            preserve_positions = (
                global_indices.remainder(PREFERENCE_COUNT) == PREFERENCE_COUNT - 1
            ).nonzero().flatten()
            if acquire_positions.numel():
                endpoint_objectives.append((
                    0, acquire[acquire_positions].sum() / question_count,
                ))
            if stage > 0 and preserve_positions.numel():
                endpoint_objectives.append((
                    1, preserve[preserve_positions].sum() / question_count,
                ))
            for position, (endpoint_index, objective) in enumerate(endpoint_objectives):
                gradients = torch.autograd.grad(
                    objective, model_parameters,
                    retain_graph=position + 1 < len(endpoint_objectives),
                    allow_unused=True,
                )
                preference_gradients[endpoint_index] = add_gradients_(
                    model_parameters, gradients,
                    preference_gradients[endpoint_index],
                )
        else:
            present = sorted(set(
                int(x) for x in global_indices.remainder(PREFERENCE_COUNT).tolist()
            ))
            for position, preference_index in enumerate(present):
                local_positions = (
                    global_indices.remainder(PREFERENCE_COUNT) == preference_index
                ).nonzero().flatten()
                objective = values[local_positions].sum() / question_count
                gradients = torch.autograd.grad(
                    objective, model_parameters,
                    retain_graph=position + 1 < len(present), allow_unused=True,
                )
                preference_gradients[preference_index] = add_gradients_(
                    model_parameters, gradients,
                    preference_gradients[preference_index],
                )

        acquire_slots = (
            global_indices.remainder(PREFERENCE_COUNT) == 0
        ).nonzero().flatten()
        preserve_slots = (
            global_indices.remainder(PREFERENCE_COUNT) == PREFERENCE_COUNT - 1
        ).nonzero().flatten()
        if acquire_slots.numel():
            endpoint_acquire_sum += float(acquire[acquire_slots].detach().sum())
        if stage > 0 and preserve_slots.numel():
            endpoint_preserve_sum += float(preserve[preserve_slots].detach().sum())
        loss_sum += float(values.detach().sum())
        acquire_sum += float(acquire.detach().sum())
        preserve_sum += 0.0 if preserve is None else float(preserve.detach().sum())
        weight_sum += float(weights.detach().sum())
        entropy_sum += float(entropy.detach().sum())
        del sliced, acquire, preserve, values, weights, entropy

    allreduce_small_grads(code_parameters)
    fast_grad_norm = torch.nn.utils.clip_grad_norm_(
        code_parameters, args.max_grad_norm,
    )

    flat_gradients = []
    for gradients in preference_gradients:
        for parameter, gradient in zip(model_parameters, gradients):
            parameter.grad = (
                torch.zeros_like(parameter) if gradient is None else gradient
            )
        flat_gradients.append(fused_allreduce_parameter_grads(
            model_parameters, dist.get_world_size(),
        ).detach())
    if args.slow_endpoint_safe_projection:
        geometry = acquire_primary_projection(
            flat_gradients[0], flat_gradients[1],
        )
        geometry.update({
            "weights": None,
            "norms": [
                math.sqrt(max(geometry["raw_norm_a_sq"], 0.0)),
                math.sqrt(max(geometry["raw_norm_p_sq"], 0.0)),
            ],
            "cosine": (
                0.0 if geometry["raw_norm_a_sq"] <= 0 or geometry["raw_norm_p_sq"] <= 0
                else geometry["raw_dot"] / math.sqrt(
                    geometry["raw_norm_a_sq"] * geometry["raw_norm_p_sq"]
                )
            ),
            "common_norm": math.sqrt(max(
                flat_norm_dot(geometry["combined"]), 0.0,
            )),
        })
    else:
        geometry = (
            four_preference_common_projection(flat_gradients)
            if args.slow_four_preference_common_projection
            else four_preference_mgda(flat_gradients)
        )
    combined = torch._utils._unflatten_dense_tensors(
        geometry["combined"], model_parameters,
    )
    for parameter, gradient in zip(model_parameters, combined):
        parameter.grad = gradient
    slow_grad_norm = torch.nn.utils.clip_grad_norm_(
        model_parameters, args.max_grad_norm,
    )

    fast_optimizer.step()
    slow_optimizer.step()
    if fast_scheduler is not None:
        fast_scheduler.step()
    slow_scheduler.step()
    fast_optimizer.zero_grad(set_to_none=True)
    slow_optimizer.zero_grad(set_to_none=True)

    if stage > 0:
        dist.all_reduce(observed_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(observed_max, op=dist.ReduceOp.MAX)
        normalization_state["minima"] = observed_min.detach().cpu()
        normalization_state["ranges"] = (
            observed_max - observed_min
        ).clamp_min(args.range_floor).detach().cpu()
    return {
        "fast_loss": loss_sum / count,
        "acquire_fkl": acquire_sum / count,
        "preserve_sft_nll": None if stage == 0 else preserve_sum / count,
        "fast_effective_acquire_weight": weight_sum / count,
        "stch_entropy": entropy_sum / count,
        "stch_min": [float(x) for x in minima],
        "stch_max": [float(x) for x in maxima],
        "stch_range": [float(x) for x in ranges],
        "normalization_delay_steps": 1,
        "fast_grad_norm": float(fast_grad_norm),
        "slow_acquire_fkl_lambda0": endpoint_acquire_sum / question_count,
        "slow_preserve_sft_nll_lambda1": (
            None if stage == 0 else endpoint_preserve_sum / question_count
        ),
        "mgda_alpha_acquire": None,
        "mgda_preference_weights": geometry["weights"],
        "mgda_preference_grad_norms": geometry["norms"],
        "mgda_preference_cosine": geometry["cosine"],
        "mgda_common_norm": geometry["common_norm"],
        "slow_preference_geometry": (
            "endpoint_acquire_projected_to_preserve_safe_halfspace_beta0"
            if args.slow_endpoint_safe_projection else (
            "common_subspace_projection"
            if args.slow_four_preference_common_projection else "mgda"
            )
        ),
        "projection_active": (
            geometry.get("active") if args.slow_endpoint_safe_projection else None
        ),
        "projection_removed_ratio": (
            geometry.get("removed_ratio")
            if args.slow_endpoint_safe_projection else None
        ),
        "preserve_dot_after_projection": (
            geometry.get("dot_after")
            if args.slow_endpoint_safe_projection else None
        ),
        "common_projection_weights": (
            geometry["weights"]
            if args.slow_four_preference_common_projection else None
        ),
        "common_projection_difference_rank": (
            geometry.get("difference_rank")
        ),
        "common_projection_removed_ratio": (
            geometry.get("removed_ratio")
        ),
        "common_projection_directional_spread": (
            geometry.get("directional_spread")
        ),
        "common_projection_directional_derivatives": (
            geometry.get("directional_derivatives")
        ),
        "slow_grad_norm_preclip": float(slow_grad_norm),
        "slow_lr": slow_scheduler.get_last_lr()[0],
        "endpoint_forward_reused": True,
    }


@torch.no_grad()
def ema_update(teacher, model, alpha):
    for target, source in zip(teacher.parameters(), model.parameters()):
        target.mul_(1.0 - alpha).add_(source, alpha=alpha)


def make_scheduler(optimizer, total_steps, warmup_steps, scheduler_type):
    def ratio(step):
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_type == "linear":
            return 1.0 - progress
        if scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, ratio)


def bucketed_epoch_order(rows, seed, epoch, bucket_size):
    """Shuffle length buckets and examples inside them to reduce padding."""
    rng = random.Random(seed + 1009 * epoch)
    buckets = {}
    for index, row in enumerate(rows):
        key = int(row["length"]) // max(1, bucket_size)
        buckets.setdefault(key, []).append(index)
    keys = list(buckets)
    rng.shuffle(keys)
    order = []
    for key in keys:
        rng.shuffle(buckets[key])
        order.extend(buckets[key])
    return order


def main():
    args = parse_args()
    if args.input_conditioned_fast and args.task_conditioned_fast:
        raise ValueError("Choose either x-conditioned or task-conditioned fast memory")
    if args.task_conditioned_fast and args.bezier_order != 3:
        raise ValueError("TaskConditionedBezierCode is cubic; use --bezier-order 3")
    if args.endpoint_transport and args.stage > 0 and not args.previous_prompt:
        raise ValueError("endpoint transport requires --previous-prompt")
    if args.preserve_objective == "topk_fkl" and args.stage > 0:
        if args.preserve_model is None or not args.preserve_model.exists():
            raise ValueError("stage > 0 topk_fkl requires --preserve-model")
    if args.training_schedule == "uniform_then_fast":
        if args.stage == 0:
            raise ValueError("uniform_then_fast requires an old task")
        if args.acquire_objective != "sft" or args.preserve_objective != "topk_fkl":
            raise ValueError(
                "uniform_then_fast requires acquire=sft and preserve=topk_fkl"
            )
        if args.fixed_acquire_scale is None or args.fixed_preserve_scale is None:
            raise ValueError("uniform_then_fast requires fixed objective scales")
    if (args.slow_four_preference_mgda
            and args.slow_four_preference_common_projection):
        raise ValueError(
            "Choose either four-preference MGDA or common projection, not both."
        )
    if ((args.slow_four_preference_mgda
         or args.slow_four_preference_common_projection)
            and args.training_schedule != "fast_only"):
        raise ValueError(
            "Four-preference slow geometry currently requires "
            "--training-schedule fast_only (the named schedule denotes one "
            "joint slow/fast phase when this flag is enabled)"
        )
    if (args.slow_endpoint_safe_projection
            and args.training_schedule != "joint_onephase"):
        raise ValueError(
            "--slow-endpoint-safe-projection requires "
            "--training-schedule joint_onephase"
        )
    if sum((
        args.mgda_fixed_scale_calibration,
        args.mgda_unit_gradient,
        args.slow_gradient_projection,
    )) > 1:
        raise ValueError(
            "Choose only one slow gradient mode: fixed-scale MGDA, "
            "unit-gradient MGDA, or acquire-primary projection."
        )
    accelerator = Accelerator(mixed_precision="bf16")
    device, rank, world = accelerator.device, accelerator.process_index, accelerator.num_processes
    if args.global_batch % world:
        raise ValueError("global batch must be divisible by world size")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.max_completion_length = args.max_completion_length or MAX_COMPLETION[args.task]
    current_rows, rejected = prepare_current_rows(args, tokenizer)
    buffer_rows = prepare_buffer_rows(args, tokenizer)
    if args.stage > 0 and not buffer_rows:
        raise RuntimeError("stage > 0 requires a non-empty replay buffer")
    local_batch = args.global_batch // world
    epochs = args.epochs or EPOCHS[args.task]
    steps = math.ceil(len(current_rows) / args.global_batch) * epochs
    baseline_rows = current_rows + buffer_rows
    baseline_steps = math.ceil(len(baseline_rows) / args.global_batch) * epochs
    if args.max_steps > 0:
        steps = min(steps, args.max_steps)
        baseline_steps = min(baseline_steps, args.max_steps)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(device)
    teacher = None
    if args.acquire_objective == "sdft":
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to(device)
        teacher.requires_grad_(False)
        teacher.eval()
        teacher.config.use_cache = False
    preserve_teacher = None
    if args.stage > 0 and args.preserve_objective == "topk_fkl":
        preserve_teacher = AutoModelForCausalLM.from_pretrained(
            args.preserve_model, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        preserve_teacher.requires_grad_(False)
        preserve_teacher.eval()
        preserve_teacher.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    model.config.use_cache = False
    if args.task_conditioned_fast:
        code_model = TaskConditionedBezierCode(
            len(TASKS), args.prompt_length, int(model.config.hidden_size),
        ).to(device)
        code_model.task_conditioned = True
    elif args.input_conditioned_fast:
        code_model = InputConditionedBezierCode(
            args.prompt_length, int(model.config.hidden_size),
            condition_width=args.condition_width,
            condition_rank=args.condition_rank,
            max_question_length=args.max_question_encoder_length,
            conditional_gate=args.condition_acquire_gate,
            bezier_order=args.bezier_order,
        ).to(device)
    else:
        code_model = CubicBezierCode(
            args.prompt_length, int(model.config.hidden_size),
        ).to(device)
    if args.stage == 0 and args.training_schedule != "fast_only":
        # Stage 0 has no previous behavior to preserve.  Keep an inert state
        # only so stage 1 can initialize its fast-memory module; it must never
        # enter the stage-0 forward graph.
        with torch.no_grad():
            code_model.controls.zero_()
    transport_source_prompt = None
    if args.previous_prompt and args.previous_prompt.exists():
        previous_state = torch.load(
            args.previous_prompt, map_location="cpu", weights_only=True,
        )
        if args.endpoint_transport and args.stage > 0:
            previous_controls = previous_state["controls"]
            # The first transported source may come from the historical
            # task-bank checkpoint [task, control, token, hidden].  Every
            # later source is a pure-lambda path [control, token, hidden].
            source_controls = (
                previous_controls[args.stage - 1]
                if previous_controls.ndim == 4 else previous_controls
            ).to(device)
            with torch.no_grad():
                target = (
                    code_model.controls[args.stage]
                    if args.task_conditioned_fast else code_model.controls
                )
                target.copy_(source_controls)
            # The frozen teacher retains the complete previous path.  At a
            # sampled lambda it is conditioned by z_{t-1}(lambda), not by a
            # single endpoint copied across the curve.
            transport_source_prompt = source_controls.detach()
        else:
            code_model.load_state_dict(previous_state)
    elif (args.stage > 0 or args.training_schedule == "fast_only") and args.fast_init == "text":
        # A raw zero vector is not a neutral soft token: it remains visible to
        # attention and shifts all subsequent positions.  Initialize every
        # Bezier control point from the same sequence of real token embeddings
        # so the initial path is shared and lies on the pretrained embedding
        # manifold.  STCH then separates the four controls during training.
        initial_ids = tokenizer(
            args.fast_init_text, add_special_tokens=False,
        ).input_ids
        if not initial_ids:
            raise ValueError("--fast-init-text tokenized to an empty sequence")
        repeats = math.ceil(args.prompt_length / len(initial_ids))
        initial_ids = (initial_ids * repeats)[:args.prompt_length]
        token_ids = torch.tensor(initial_ids, device=device, dtype=torch.long)
        initial_prompt = model.get_input_embeddings()(token_ids).detach()
        with torch.no_grad():
            if args.task_conditioned_fast:
                code_model.controls.copy_(
                    initial_prompt.view(1, 1, args.prompt_length, -1)
                    .expand_as(code_model.controls)
                )
            else:
                code_model.controls.copy_(
                    initial_prompt.unsqueeze(0).expand_as(code_model.controls)
                )

    slow_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.slow_lr, weight_decay=0.0, fused=True,
    )
    fast_optimizer = torch.optim.AdamW(
        [p for p in code_model.parameters() if p.requires_grad],
        lr=args.fast_lr, weight_decay=0.0, fused=True,
    )
    slow_schedule_steps = (
        baseline_steps if args.training_schedule == "baseline_then_fast" else steps
    )
    scheduler = make_scheduler(
        slow_optimizer, slow_schedule_steps,
        min(args.warmup_steps, slow_schedule_steps),
        args.lr_scheduler_type,
    )
    # Keep the conditional memory synchronized with the final slow backbone.
    # The historical schedule froze fast memory for the last 10% while slow
    # continued moving, leaving evaluation at (theta_final, z_0.9T).  For all
    # continual stages, update slow first and then fast for every step.
    fast_active_steps = 0 if (
        args.stage == 0 and args.training_schedule != "fast_only"
    ) else (
        2 * steps if args.training_schedule == "joint_random_then_fast" else steps
    )
    fast_scheduler = (
        None if args.stage == 0 else
        make_scheduler(fast_optimizer, fast_active_steps, 0, "linear")
    )

    required = steps * args.global_batch
    order, epoch = [], 0
    while len(order) < required:
        values = bucketed_epoch_order(
            current_rows, args.seed, epoch, args.length_bucket_size,
        )
        order.extend(values)
        epoch += 1
    baseline_required = baseline_steps * args.global_batch
    baseline_order, baseline_epoch = [], 0
    while len(baseline_order) < baseline_required:
        values = bucketed_epoch_order(
            baseline_rows, args.seed, baseline_epoch, args.length_bucket_size,
        )
        baseline_order.extend(values)
        baseline_epoch += 1
    replay_cursor = 0
    normalization_state = {
        "minima": torch.zeros(2), "ranges": torch.ones(2),
    }
    mgda_calibration_state = {
        "acquire_scale": None, "preserve_scale": None,
    }
    dwa_state = {"acquire_losses": [], "preserve_losses": []}
    start_time = time.monotonic()
    if rank == 0:
        config = vars(args).copy()
        config.update({
            "event": "config", "method": "LAPS",
            "train_rows": len(current_rows), "overlength_rejected": rejected,
            "buffer_rows": len(buffer_rows), "steps": steps,
            "baseline_slow_rows": len(baseline_rows),
            "baseline_slow_steps": baseline_steps,
            "global_batch": args.global_batch,
            "world_size": world,
            "per_rank_batch": local_batch,
            "preferences": (
                [] if args.stage == 0 else (
                    ["batch_shared_uniform_1", "batch_shared_uniform_2",
                     "batch_shared_uniform_3", "batch_shared_uniform_4"]
                    if args.slow_four_preference_common_projection else (
                    ["batch_shared_uniform_1", "batch_shared_uniform_2",
                     "batch_shared_uniform_3", "batch_shared_uniform_4"]
                    if (args.slow_four_preference_mgda
                        or args.batch_shared_random_preferences) else
                    [0.0, "uniform_random_1", "uniform_random_2", 1.0]
                ))
            ),
            "fast_initialization": (
                "inert-stage0" if (
                    args.stage == 0 and args.training_schedule != "fast_only"
                ) else
                "previous-checkpoint" if args.previous_prompt else args.fast_init
            ),
            "preference_count": PREFERENCE_COUNT,
            "soft_prompt_placement": "before_assistant",
            "fast_objective": (
                "disabled; exact plain SFT degeneration"
                if (args.stage == 0 and args.training_schedule != "fast_only") else (
                (
                    "STCH(acquire gold-response SFT-NLL, preserve old-checkpoint top-64+OTHER FKL)"
                    if args.preserve_objective == "topk_fkl" else
                    "STCH(acquire gold-response SFT-NLL, preserve replay SFT)"
                )
                if args.acquire_objective == "sft" else
                f"STCH(acquire SDFT-{args.acquire_kl_direction.upper()}-KL, preserve OPR-SFT)"
                )
            ),
            "fast_scalarization": args.fast_scalarization,
            "input_conditioned_fast": args.input_conditioned_fast,
            "task_conditioned_fast": args.task_conditioned_fast,
            "preserve_objective": args.preserve_objective,
            "preserve_model": str(args.preserve_model) if args.preserve_model else None,
            "preserve_top_k": args.preserve_top_k,
            "endpoint_transport": args.endpoint_transport,
            "transport_source_task": (
                TASKS[args.stage - 1]
                if args.endpoint_transport and args.stage > 0 else None
            ),
            "condition_width": (
                args.condition_width if args.input_conditioned_fast else None
            ),
            "condition_rank": (
                args.condition_rank if args.input_conditioned_fast else None
            ),
            "condition_acquire_gate": (
                args.condition_acquire_gate if args.input_conditioned_fast else None
            ),
            "acquire_objective": args.acquire_objective,
            "acquire_kl_direction": args.acquire_kl_direction,
            "slow_objective": (
                "fixed lambda=0.5 STCH with fixed functional scales"
                if args.training_schedule == "uniform_then_fast" else (
                "four-preference common-subspace projection over STCH scalarizations"
                if args.slow_four_preference_common_projection else (
                "four-preference raw-gradient MGDA over STCH scalarizations"
                if args.slow_four_preference_mgda else (
                "exact ordinary SFT on concatenated current+replay data"
                if args.training_schedule == "baseline_then_fast" else (
                "fixed-scale-normalized Dynamic Weight Averaging"
                if args.slow_dwa else (
                "acquire-primary preserve-safe gradient projection"
                if args.slow_gradient_projection else (
                    "two-endpoint per-step unit-gradient MGDA"
                    if args.mgda_unit_gradient else (
                    "two-endpoint fixed-scale-calibrated MGDA"
                    if args.mgda_fixed_scale_calibration
                    else "two-endpoint raw-gradient MGDA"
                    ))
                )))))
            ),
            "mgda_scale_calibration": (
                "not applicable; preserve is a half-space constraint"
                if args.slow_gradient_projection else (
                    "per-step unit norms; acquire norm restores common magnitude"
                    if args.mgda_unit_gradient else (
                    "first distributed training batch; fixed for stage"
                    if args.mgda_fixed_scale_calibration else "disabled"
                    )
                )
            ),
            "normalization": (
                f"fixed functional nats/token scales "
                f"[{args.fixed_acquire_scale}, {args.fixed_preserve_scale}]"
                if args.fixed_acquire_scale is not None else (
                "one-step-delayed batch min/range"
                if args.stch_normalization and args.fast_scalarization == "stch"
                else "none (raw objectives)"
                )
            ),
            "endpoint_reuse": (
                "same forward graphs reused by slow four-preference MGDA and fast STCH"
                if args.slow_four_preference_mgda else
                "disabled; plain slow update followed by fresh fast forward"
            ),
            "training_schedule": args.training_schedule,
            "update_order": (
                "stage0 plain slow only" if (
                    args.stage == 0 and args.training_schedule != "fast_only"
                ) else
                (
                    "fixed-uniform slow phase, then random-four-preference fast phase"
                    if args.training_schedule == "uniform_then_fast" else (
                    "simultaneous slow four-preference MGDA and fast STCH"
                    if args.slow_four_preference_mgda else
                    (
                    "task-level two-phase: joint random-lambda DWA/projection, "
                    "then conditional fast STCH"
                    if args.training_schedule == "joint_random_then_fast" else
                    (
                        "exact SFT+Replay baseline on concatenated data, then conditional fast STCH"
                        if args.training_schedule == "baseline_then_fast" else
                        "task-level two-phase: all plain slow steps, then all conditional fast steps"
                    )))
                )
            ),
            "fast_active_steps": fast_active_steps,
            "fast_freeze_fraction": 0.0 if args.stage > 0 else None,
            "mgda_collective": "fused flattened all-reduce",
            "pretokenized": True,
            "length_bucket_size": args.length_bucket_size,
        })
        print(json.dumps(config, default=str), flush=True)

    def step_tensors(step, single_random_preference=False,
                     fixed_uniform_preference=False):
        begin = (step - 1) * args.global_batch + rank * local_batch
        indices = order[begin:begin + local_batch]
        rows = [current_rows[index] for index in indices]
        if buffer_rows:
            replay = [
                buffer_rows[((step - 1) * args.global_batch + rank * local_batch + i)
                            % len(buffer_rows)]
                for i in range(local_batch)
            ]
        else:
            replay = []
        if fixed_uniform_preference:
            lambdas = torch.full(
                (local_batch,), 0.5, device=device, dtype=torch.float32,
            )
        elif single_random_preference:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(args.seed + args.stage * 100_003 + step)
            sampled = float(torch.rand((), generator=generator))
            lambdas = torch.full(
                (local_batch,), sampled, device=device, dtype=torch.float32,
            )
        else:
            lambdas = preference_values(
                local_batch, step, args.seed, device, rank, world,
                batch_shared=(args.slow_four_preference_mgda
                              or args.slow_four_preference_common_projection
                              or args.batch_shared_random_preferences),
                symmetric_endpoints=False,
            )
        tensors = build_step_tensors(
            rows, replay, lambdas, tokenizer, args, device,
        )
        tensors["constant_preserve_target_cache"] = (
            attach_constant_preserve_targets(
                tensors, preserve_teacher, transport_source_prompt, args,
            )
            if args.preserve_objective == "topk_fkl" else False
        )
        if args.endpoint_transport and replay:
            # Both objectives must act on the same current-stage curve.  The
            # replay task identity remains in the text protocol, while its
            # fast-memory operating point is the current continual stage.
            tensors["replay_task_ids"].fill_(args.stage)
        if args.input_conditioned_fast:
            pref_count = tensors["preference_count"]
            current_features = code_model.encode_questions(
                [str(row.get("prompt", row["prompt_text"])) for row in rows],
                device,
            )
            replay_features = code_model.encode_questions(
                [str(row.get("prompt", row["prompt_text"])) for row in replay],
                device,
            )
            tensors["question_features"] = current_features.repeat_interleave(
                pref_count, dim=0,
            )
            tensors["replay_question_features"] = replay_features.repeat_interleave(
                pref_count, dim=0,
            )
        return lambdas, tensors

    def baseline_step_tensors(step):
        """One ordinary SFT example per item from current+replay mixture."""
        begin = (step - 1) * args.global_batch + rank * local_batch
        indices = baseline_order[begin:begin + local_batch]
        rows = [baseline_rows[index] for index in indices]
        lambdas = torch.zeros(local_batch, device=device, dtype=torch.float32)
        return lambdas, build_step_tensors(
            rows, [], lambdas, tokenizer, args, device,
        )

    if args.stage == 0 and args.training_schedule != "fast_only":
      for step in range(1, steps + 1):
        lambdas, tensors = step_tensors(step)
        completion_mask = tensors["current_target_mask"]
        update_metrics = stage0_plain_sft_update(
            model, tensors, slow_optimizer, scheduler, args,
        )
        if rank == 0:
            print(json.dumps({
                "event": "step", "phase": "slow", "stage": args.stage,
                "task": args.task, "step": step, "steps": steps,
                "optimization_step": step, "optimization_steps": steps,
                "effective_epoch": step * args.global_batch / len(current_rows),
                "random_lambda_means": None,
                "soft_prompt_norm": float(code_model.controls.detach().float().norm()),
                "completion_length_mean": float(completion_mask.sum(-1).float().mean()),
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "elapsed_s": time.monotonic() - start_time,
                **update_metrics,
            }), flush=True)
        torch.cuda.reset_peak_memory_stats(device)
    else:
      if args.training_schedule == "joint_onephase":
        # Every iteration updates slow first and fast immediately afterwards
        # on the same shuffled current/replay batch.  There is no task-level
        # Phase A/Phase B boundary.
        for step in range(1, steps + 1):
            lambdas, tensors = step_tensors(step)
            completions = tensors["current_target_ids"]
            completion_mask = tensors["current_target_mask"]
            if args.slow_endpoint_safe_projection:
                joint_metrics = joint_four_preference_mgda_stch_update(
                    model, teacher, code_model, tensors, completions,
                    completion_mask, fast_optimizer, slow_optimizer,
                    scheduler, fast_scheduler, args, args.stage,
                    normalization_state,
                )
                if rank == 0:
                    print(json.dumps({
                        "event": "step",
                        "phase": "joint_endpoint_safe_projection_stch",
                        "stage": args.stage, "task": args.task,
                        "step": step, "steps": steps,
                        "optimization_step": step,
                        "optimization_steps": steps,
                        "effective_epoch": (
                            step * args.global_batch / len(current_rows)
                        ),
                        "sampled_lambdas": [
                            float(x) for x in lambdas.view(
                                -1, PREFERENCE_COUNT,
                            )[0]
                        ],
                        "soft_prompt_norm": float(
                            code_model.controls.detach().float().norm()
                        ),
                        "completion_length_mean": float(
                            completion_mask.sum(-1).float().mean()
                        ),
                        "peak_memory_gib": (
                            torch.cuda.max_memory_allocated(device) / 2**30
                        ),
                        "elapsed_s": time.monotonic() - start_time,
                        "bezier_order": args.bezier_order,
                        "prompt_length": args.prompt_length,
                        "slow_projection_beta": 0.0,
                        **joint_metrics,
                    }), flush=True)
                torch.cuda.reset_peak_memory_stats(device)
                continue
            if args.onephase_slow_mode == "acquire_sft":
                raw_slow = stage0_plain_sft_update(
                    model, tensors, slow_optimizer, scheduler, args,
                )
                slow_metrics = {
                    "slow_acquire_fkl_lambda0": raw_slow["slow_acquire_fkl_lambda0"],
                    "slow_preserve_sft_nll_lambda1": None,
                    "grad_norm_acquire": raw_slow["grad_norm_acquire"],
                    "grad_norm_preserve": 0.0,
                    "gradient_cosine": 0.0,
                    "projection_active": None,
                    "projection_removed_ratio": None,
                    "slow_grad_norm_preclip": raw_slow["slow_grad_norm_preclip"],
                    "slow_lr": raw_slow["slow_lr"],
                }
            else:
                slow_metrics = slow_plain_projection_update(
                    model, tensors, slow_optimizer, scheduler, args,
                    mgda_calibration_state, dwa_state,
                )
            fast_metrics = fast_update(
                model, teacher, code_model, tensors, completions,
                completion_mask, fast_optimizer, fast_scheduler, args,
                args.stage, normalization_state, preserve_teacher,
                transport_source_prompt,
            )
            fast_metrics["endpoint_forward_reused"] = False
            if rank == 0:
                print(json.dumps({
                    "event": "step", "phase": "joint_onephase",
                    "stage": args.stage, "task": args.task,
                    "step": step, "steps": steps,
                    "optimization_step": step,
                    "optimization_steps": steps,
                    "effective_epoch": step * args.global_batch / len(current_rows),
                    "random_lambda_means": [
                        float(x) for x in
                        lambdas.view(-1, PREFERENCE_COUNT)[:, 1:3].mean(0)
                    ],
                    "soft_prompt_norm": float(
                        code_model.controls.detach().float().norm()
                    ),
                    "completion_length_mean": float(
                        completion_mask.sum(-1).float().mean()
                    ),
                    "peak_memory_gib": (
                        torch.cuda.max_memory_allocated(device) / 2**30
                    ),
                    "elapsed_s": time.monotonic() - start_time,
                    "onephase_slow_mode": args.onephase_slow_mode,
                    **slow_metrics, **fast_metrics,
                }), flush=True)
            torch.cuda.reset_peak_memory_stats(device)
      elif args.training_schedule == "joint_random_then_fast":
        if args.acquire_objective != "sft":
            raise ValueError(
                "joint_random_then_fast currently requires --acquire-objective sft"
            )
        # Phase A: one batch-shared random preference, with conflict geometry
        # applied to the concatenated theta/phi gradients.
        for step in range(1, steps + 1):
            lambdas, tensors = step_tensors(
                step, single_random_preference=True,
            )
            completions = tensors["current_target_ids"]
            completion_mask = tensors["current_target_mask"]
            joint_metrics = joint_random_preference_update(
                model, code_model, tensors, completions, completion_mask,
                slow_optimizer, fast_optimizer, scheduler, fast_scheduler,
                args, mgda_calibration_state, dwa_state,
            )
            if rank == 0:
                print(json.dumps({
                    "event": "step", "phase": "joint_random",
                    "stage": args.stage, "task": args.task,
                    "step": step, "steps": steps,
                    "optimization_step": step,
                    "optimization_steps": 2 * steps,
                    "effective_epoch": step * args.global_batch / len(current_rows),
                    "soft_prompt_norm": float(
                        code_model.controls.detach().float().norm()
                    ),
                    "completion_length_mean": float(
                        completion_mask.sum(-1).float().mean()
                    ),
                    "peak_memory_gib": (
                        torch.cuda.max_memory_allocated(device) / 2**30
                    ),
                    "elapsed_s": time.monotonic() - start_time,
                    **joint_metrics,
                }), flush=True)
            torch.cuda.reset_peak_memory_stats(device)
      elif args.training_schedule == "baseline_then_fast":
        # Phase A: exactly reproduce SFT+Replay.  Replay is mixed into the
        # dataset at its natural buffer frequency; no preference prompt and
        # no multi-objective geometry enter this graph.
        code_model.requires_grad_(False)
        for step in range(1, baseline_steps + 1):
            lambdas, tensors = baseline_step_tensors(step)
            completion_mask = tensors["current_target_mask"]
            slow_metrics = stage0_plain_sft_update(
                model, tensors, slow_optimizer, scheduler, args,
            )
            slow_metrics["slow_prompt_conditioning"] = "none"
            slow_metrics["slow_data_policy"] = "current_plus_replay_concatenation"
            if rank == 0:
                print(json.dumps({
                    "event": "step", "phase": "baseline_slow",
                    "stage": args.stage, "task": args.task,
                    "step": step, "steps": baseline_steps,
                    "optimization_step": step,
                    "optimization_steps": baseline_steps + steps,
                    "effective_epoch": (
                        step * args.global_batch / len(baseline_rows)
                    ),
                    "soft_prompt_norm": float(
                        code_model.controls.detach().float().norm()
                    ),
                    "completion_length_mean": float(
                        completion_mask.sum(-1).float().mean()
                    ),
                    "peak_memory_gib": (
                        torch.cuda.max_memory_allocated(device) / 2**30
                    ),
                    "elapsed_s": time.monotonic() - start_time,
                    **slow_metrics,
                }), flush=True)
            torch.cuda.reset_peak_memory_stats(device)
      elif args.training_schedule == "task_twophase":
        # Phase A: establish the complete task's slow backbone without any
        # soft prompt in the forward graph.
        code_model.requires_grad_(False)
        for step in range(1, steps + 1):
            lambdas, tensors = step_tensors(step)
            slow_metrics = slow_plain_projection_update(
                model, tensors, slow_optimizer, scheduler, args,
                mgda_calibration_state, dwa_state,
            )
            if teacher is not None:
                ema_update(teacher, model, args.ema_alpha)
            if rank == 0:
                print(json.dumps({
                    "event": "step", "phase": "slow", "stage": args.stage,
                    "task": args.task, "step": step, "steps": steps,
                    "optimization_step": step, "optimization_steps": 2 * steps,
                    "effective_epoch": step * args.global_batch / len(current_rows),
                    "random_lambda_means": None,
                    "soft_prompt_norm": float(code_model.controls.detach().float().norm()),
                    "completion_length_mean": float(
                        tensors["current_target_mask"].sum(-1).float().mean()
                    ),
                    "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "elapsed_s": time.monotonic() - start_time,
                    **slow_metrics,
                }), flush=True)
            torch.cuda.reset_peak_memory_stats(device)

      elif args.training_schedule == "uniform_then_fast":
        # Phase A: fixed midpoint operating point. Fast memory defines the
        # coordinate system but is frozen; only the slow backbone changes.
        code_model.requires_grad_(False)
        for step in range(1, steps + 1):
            lambdas, tensors = step_tensors(
                step, fixed_uniform_preference=True,
            )
            slow_metrics = fixed_uniform_slow_update(
                model, code_model, tensors, slow_optimizer, scheduler, args,
                preserve_teacher, transport_source_prompt,
            )
            if rank == 0:
                print(json.dumps({
                    "event": "step", "phase": "slow_fixed_uniform",
                    "stage": args.stage, "task": args.task,
                    "step": step, "steps": steps,
                    "optimization_step": step,
                    "optimization_steps": 2 * steps,
                    "effective_epoch": step * args.global_batch / len(current_rows),
                    "soft_prompt_norm": float(code_model.controls.detach().float().norm()),
                    "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "elapsed_s": time.monotonic() - start_time,
                    **slow_metrics,
                }), flush=True)
            torch.cuda.reset_peak_memory_stats(device)

      elif args.training_schedule != "fast_only":
        raise ValueError(f"unsupported training schedule: {args.training_schedule}")

      # Phase B: freeze the now-final slow backbone and fit the full fast
      # preference path on the same shuffled task/replay sequence.
      model.requires_grad_(False)
      if args.gradient_checkpointing:
          # Theta is frozen in Phase B, so activation checkpointing only
          # recomputes the backbone while differentiating w.r.t. prompt input.
          # Disabling it changes neither logits nor gradients.
          model.gradient_checkpointing_disable()
      for step in (
          () if args.training_schedule == "joint_onephase"
          else range(1, steps + 1)
      ):
        lambdas, tensors = step_tensors(step)
        if args.acquire_objective == "sft":
            # Gold targets are the acquisition trajectory; no rollout or
            # privileged teacher model is needed.
            completions = tensors["current_target_ids"]
            completion_mask = tensors["current_target_mask"]
        else:
            model.requires_grad_(False)
            code_model.requires_grad_(False)
            completions, completion_mask = generate_current(
                model, code_model, tensors["prompt_ids"], tensors["prompt_mask"],
                tensors["positions"], lambdas, tokenizer, args,
            )
        if (args.slow_four_preference_mgda
                or args.slow_four_preference_common_projection):
            fast_metrics = joint_four_preference_mgda_stch_update(
                model, teacher, code_model, tensors, completions,
                completion_mask, fast_optimizer, slow_optimizer,
                scheduler, fast_scheduler, args, args.stage,
                normalization_state,
            )
        else:
            fast_metrics = fast_update(
                model, teacher, code_model, tensors, completions,
                completion_mask, fast_optimizer, fast_scheduler, args,
                args.stage, normalization_state, preserve_teacher,
                transport_source_prompt,
            )
            fast_metrics["endpoint_forward_reused"] = False
        if rank == 0:
            print(json.dumps({
                "event": "step",
                "phase": (
                    "joint_4pref_common_projection_stch"
                    if args.slow_four_preference_common_projection else (
                        "joint_4pref_mgda_stch"
                        if args.slow_four_preference_mgda else "fast"
                    )
                ),
                "stage": args.stage,
                "task": args.task,
                "step": step, "steps": steps,
                "optimization_step": (
                    (baseline_steps if args.training_schedule == "baseline_then_fast" else (
                        0 if args.training_schedule == "fast_only" else steps
                    ))
                    + step
                ),
                "optimization_steps": (
                    (baseline_steps if args.training_schedule == "baseline_then_fast" else (
                        0 if args.training_schedule == "fast_only" else steps
                    ))
                    + steps
                ),
                "effective_epoch": step * args.global_batch / len(current_rows),
                "random_lambda_means": (
                    None if args.stage == 0 else [
                        float(x) for x in
                        lambdas.view(-1, PREFERENCE_COUNT)[:, 1:3].mean(0)
                    ]
                ),
                "sampled_lambdas": (
                    [float(x) for x in lambdas.view(
                        -1, PREFERENCE_COUNT,
                    )[0]]
                    if (args.slow_four_preference_mgda
                        or args.slow_four_preference_common_projection
                        or args.batch_shared_random_preferences) else None
                ),
                "soft_prompt_norm": float(code_model.controls.detach().float().norm()),
                "completion_length_mean": float(completion_mask.sum(-1).float().mean()),
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "elapsed_s": time.monotonic() - start_time,
                **fast_metrics,
            }), flush=True)
        torch.cuda.reset_peak_memory_stats(device)

    accelerator.wait_for_everyone()
    if rank == 0:
        if args.save_frozen_model or args.training_schedule != "fast_only":
            model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(code_model.state_dict(), args.output_dir / "preference_soft_prompt.pt")
        (args.output_dir / "preference_config.json").write_text(json.dumps({
            "method": "LAPS", "prompt_length": args.prompt_length,
            "bezier_order": args.bezier_order,
            "lambda_semantics": {"0": "acquire", "1": "preserve"},
            "stage": args.stage, "task": args.task,
            "input_conditioned": args.input_conditioned_fast,
            "task_conditioned": args.task_conditioned_fast,
            "num_tasks": len(TASKS) if args.task_conditioned_fast else None,
            "slow_checkpoint": str(args.model),
            "preserve_objective": args.preserve_objective,
            "endpoint_transport": args.endpoint_transport,
            "stage_conditioned_path": args.endpoint_transport,
            "soft_prompt_placement": "before_assistant",
            "condition_width": (
                args.condition_width if args.input_conditioned_fast else None
            ),
            "condition_rank": (
                args.condition_rank if args.input_conditioned_fast else None
            ),
            "condition_acquire_gate": (
                args.condition_acquire_gate if args.input_conditioned_fast else None
            ),
            "max_question_encoder_length": (
                args.max_question_encoder_length
                if args.input_conditioned_fast else None
            ),
            "fast_scalarization": args.fast_scalarization,
        }, indent=2) + "\n")
        (args.output_dir / "STAGE_COMPLETE").touch()
        print(json.dumps({
            "event": "stage_complete", "stage": args.stage,
            "task": args.task, "output": str(args.output_dir),
            "elapsed_s": time.monotonic() - start_time,
        }), flush=True)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
