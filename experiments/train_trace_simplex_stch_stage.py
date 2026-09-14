#!/usr/bin/env python3
"""Two-phase training of an incremental cubic Bezier preference simplex.

At a new continual stage, the previous simplex is embedded as the old face and
new mixed controls are introduced.  Phase A jointly trains the slow backbone
and only those new controls at the uniform simplex barycenter.  Phase B freezes
the backbone and trains the same new controls at four random Dirichlet
preferences.  Old-face controls never move, so the previous preference chart
is retained exactly in fast-memory coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from src.simplex_bezier_prompt import (  # noqa: E402
    SimplexBezierPrompt, simplex_multi_indices, verify_recursive_expansion,
)
from src.preference_memory import prompt_position  # noqa: E402
from train_trace_opr_stch_mgda_stage import (  # noqa: E402
    EPOCHS,
    MAX_COMPLETION,
    TASKS,
    bucketed_epoch_order,
    completion_logits,
    make_scheduler,
    pad_id_rows,
    plain_completion_logits,
    prepare_baseline_sft_rows,
    prepare_buffer_rows,
    prepare_current_rows,
    sft_nll_per_example,
)
from src.distillation import (  # noqa: E402
    teacher_topk_tail_targets,
    topk_tail_forward_kl_from_targets,
)


class LastQuarterMemoryReader(nn.Module):
    """Compatibility guard for the unpublished hidden-residual ablation."""

    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(
            "This release implements LAPS with soft prompts. "
            "The legacy hidden-residual ablation is intentionally excluded."
        )


# Keep the original token-insertion implementation available for legacy runs.
_token_insertion_completion_logits = completion_logits
_ACTIVE_HIDDEN_RESIDUAL = None


def _source_signature(path):
    """Cheap cache invalidation without rereading a multi-megabyte dataset."""
    path = Path(path)
    if not path.exists():
        return None
    stat = path.stat()
    return [str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)]


def _prepared_rows_cache_key(args, tokenizer):
    buffer_manifest = (
        args.buffer.with_name("buffer.manifest.json")
        if args.buffer is not None else None
    )
    payload = {
        "version": 2,
        "task": args.task,
        "stage": int(args.stage),
        "seed": int(args.seed),
        "max_train_samples": int(args.max_train_samples),
        "max_prompt_length": int(args.max_prompt_length),
        "max_completion_length": int(args.max_completion_length),
        "acquire_objective": str(args.acquire_objective),
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": int(len(tokenizer)),
        "chat_template": tokenizer.chat_template,
        "train_source": _source_signature(args.data_root / args.task / "train.json"),
        "buffer_source": _source_signature(args.buffer) if args.buffer else None,
        "buffer_manifest": (
            _source_signature(buffer_manifest) if buffer_manifest else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def load_cached_prepared_rows(args, tokenizer, accelerator):
    """Tokenize once on rank 0 and reuse the exact rows on every DDP rank."""
    cache_dir = Path(args.data_root) / ".laps_token_cache"
    cache_key = _prepared_rows_cache_key(args, tokenizer)
    cache_path = cache_dir / f"{args.task}_stage{args.stage}_{cache_key}.json"
    if accelerator.is_main_process and not cache_path.exists():
        started = time.perf_counter()
        current_rows, rejected = prepare_current_rows(args, tokenizer)
        replay_rows = prepare_buffer_rows(args, tokenizer)
        current_rows, current_sft_dropped = prepare_baseline_sft_rows(
            current_rows, tokenizer, max_length=args.max_prompt_length,
        )
        replay_rows, replay_sft_dropped = prepare_baseline_sft_rows(
            replay_rows, tokenizer, max_length=args.max_prompt_length,
        )
        rejected += current_sft_dropped
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(
            f".tmp.{os.getpid()}.json"
        )
        temporary.write_text(json.dumps({
            "current_rows": current_rows,
            "rejected": rejected,
            "replay_rows": replay_rows,
            "current_sft_dropped": current_sft_dropped,
            "replay_sft_dropped": replay_sft_dropped,
        }, ensure_ascii=False))
        os.replace(temporary, cache_path)
        print(json.dumps({
            "event": "token_cache_build",
            "path": str(cache_path),
            "elapsed_s": time.perf_counter() - started,
            "current_rows": len(current_rows),
            "replay_rows": len(replay_rows),
            "current_sft_dropped": current_sft_dropped,
            "replay_sft_dropped": replay_sft_dropped,
        }), flush=True)
    accelerator.wait_for_everyone()
    started = time.perf_counter()
    cached = json.loads(cache_path.read_text())
    if accelerator.is_main_process:
        print(json.dumps({
            "event": "token_cache_load",
            "path": str(cache_path),
            "elapsed_s": time.perf_counter() - started,
        }), flush=True)
    return (
        cached["current_rows"], int(cached["rejected"]),
        cached["replay_rows"],
    )


class SimplexHiddenResidual(nn.Module):
    """Bezier memory plus a last-quarter, exactly gateable residual reader."""

    def __init__(self, model, num_tasks, degree, memory_tokens, scale_init,
                 o_lora_rank, layer_specific_controls=False,
                 trainable_layer_alpha=False, current_slow_anchor=False):
        super().__init__()
        reader_count = max(1, math.ceil(len(model.model.layers) * 0.25))
        self.layer_specific_controls = bool(layer_specific_controls)
        self.generator = (
            LayerSpecificSimplexBezierMemory(
                num_tasks, degree, reader_count, memory_tokens,
                int(model.config.hidden_size),
            ) if self.layer_specific_controls else
            SimplexBezierPrompt(
                num_tasks, degree, memory_tokens, int(model.config.hidden_size),
            )
        )
        # Break the all-zero preference symmetry without inheriting arbitrary
        # natural-language token embeddings.  The final coordinate is reserved
        # for request-specific alpha transport in vLLM and is always zero here.
        with torch.no_grad():
            self.generator.controls.normal_(mean=0.0, std=0.02)
            self.generator.controls[..., -1].zero_()
        self.reader = LastQuarterMemoryReader(
            model, fraction=0.25, o_lora_rank=o_lora_rank,
            # Keep R linear in request amplitude. Otherwise an RMS
            # normalization after alpha-scaled V would cancel every non-zero
            # alpha at request-specific vLLM inference.
            normalize_residual=False, scale_init=scale_init,
        )
        self.trainable_layer_alpha = bool(trainable_layer_alpha)
        self.current_slow_anchor = bool(current_slow_anchor)
        if self.trainable_layer_alpha:
            initial_alpha = 0.5
            self.layer_alpha_logits = nn.Parameter(torch.full(
                (reader_count,), math.log(initial_alpha / (1.0 - initial_alpha)),
                dtype=torch.float32,
            ))
        else:
            # No separate a_l amplitude: the reader's own s_l is the single
            # trainable per-layer residual scale.
            self.register_parameter("layer_alpha_logits", None)

    def layer_alphas(self):
        if self.layer_alpha_logits is None:
            return None
        return torch.sigmoid(self.layer_alpha_logits)

    @property
    def controls(self):
        return self.generator.controls

    @property
    def multi_indices(self):
        return self.generator.multi_indices

    @property
    def num_tasks(self):
        return self.generator.num_tasks

    @property
    def residual_scale(self):
        return self.generator.residual_scale

    @residual_scale.setter
    def residual_scale(self, value):
        self.generator.residual_scale = float(value)

    def forward(self, preferences, *, isolate_vertex_gradients=False):
        memory = self.generator(
            preferences,
            isolate_vertex_gradients=isolate_vertex_gradients,
        )
        # The last hidden coordinate transports a per-request amplitude to the
        # hook without adding tokens to the main sequence.  At the newest-task
        # simplex vertex, 1-lambda_current is exactly zero, so every memory
        # branch is an exact no-op and the policy is strictly Slow-only.
        if self.current_slow_anchor:
            gate = (1.0 - preferences[:, -1]).to(memory.dtype)
            gate = gate.view(
                gate.shape[0], *([1] * (memory.ndim - 2)), 1,
            ).expand_as(memory[..., -1:])
        else:
            gate = torch.zeros_like(memory[..., -1:])
        return torch.cat((memory[..., :-1], gate), -1)

    def metadata(self):
        return {
            **self.generator.metadata(),
            "fast_channel": "last_quarter_cross_attention_hidden_residual",
            "reader_layers": list(self.reader.layer_indices),
            "reader_o_lora_rank": int(self.reader.o_lora_rank),
            "reader_scale_init": float(self.reader.scale_init),
            "normalize_reader_residual": False,
            "request_specific_alpha": self.trainable_layer_alpha,
            "layer_specific_controls": self.layer_specific_controls,
            "trainable_layer_alpha": self.trainable_layer_alpha,
            "current_slow_anchor": self.current_slow_anchor,
            "layer_alphas": (
                self.layer_alphas().detach().cpu().tolist()
                if self.layer_alphas() is not None else None
            ),
        }

    def requires_grad_(self, requires_grad=True):
        # Do not accidentally unfreeze the copied q/k/v/o coordinate system.
        self.generator.requires_grad_(requires_grad)
        for name, parameter in self.reader.named_parameters():
            # When a_l is learned, keep the reader scale fixed so the product
            # a_l*s_l is identifiable rather than two redundant amplitudes.
            trainable = (
                (name == "layer_scales" and not self.trainable_layer_alpha)
                or name.endswith("lora_a") or name.endswith("lora_b")
            )
            parameter.requires_grad_(bool(requires_grad and trainable))
        if self.layer_alpha_logits is not None:
            self.layer_alpha_logits.requires_grad_(bool(requires_grad))
        return self


class LayerSpecificSimplexBezierMemory(nn.Module):
    """A separate cubic Bezier simplex P_{l,alpha} for every reader layer."""

    def __init__(self, num_tasks, degree, num_layers, prompt_length, hidden_size):
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.degree = int(degree)
        self.num_layers = int(num_layers)
        self.prompt_length = int(prompt_length)
        self.hidden_size = int(hidden_size)
        self.residual_scale = 1.0
        indices = simplex_multi_indices(self.num_tasks, self.degree)
        self.register_buffer(
            "multi_indices", torch.tensor(indices, dtype=torch.long),
            persistent=True,
        )
        # Control dimension stays first so existing old-face gradient masks and
        # diagnostics remain correct: [control, layer, token, hidden].
        self.controls = nn.Parameter(torch.zeros(
            len(indices), self.num_layers, self.prompt_length, self.hidden_size,
            dtype=torch.float32,
        ))
        numerator = math.factorial(self.degree)
        coefficients = [
            numerator / math.prod(math.factorial(v) for v in alpha)
            for alpha in indices
        ]
        self.register_buffer(
            "multinomial_coefficients",
            torch.tensor(coefficients, dtype=torch.float32), persistent=True,
        )

    def basis(self, preferences):
        values = preferences.float().clamp_min(0)
        values = values / values.sum(-1, keepdim=True).clamp_min(1e-12)
        terms = values.unsqueeze(1).pow(
            self.multi_indices.to(values.device).unsqueeze(0)
        ).prod(-1)
        return terms * self.multinomial_coefficients.to(values.device)

    def forward(self, preferences, *, isolate_vertex_gradients=False):
        basis = self.basis(preferences).to(self.controls.dtype)
        if isolate_vertex_gradients:
            vertex_task = self.multi_indices.eq(self.degree).float().argmax(-1)
            is_vertex_control = self.multi_indices.eq(self.degree).any(-1)
            exact_vertex = preferences.float().ge(1.0 - 1e-6)
            live = (
                ~is_vertex_control.unsqueeze(0)
                | exact_vertex[:, vertex_task].to(is_vertex_control.device)
            ).to(basis.dtype)
            effective_basis = (
                self.residual_scale * basis
                + (1.0 - self.residual_scale) / basis.shape[-1]
            )
            memory = torch.einsum(
                "bc,clth->blth", effective_basis * live, self.controls,
            ) + torch.einsum(
                "bc,clth->blth", effective_basis * (1.0 - live),
                self.controls.detach(),
            )
        else:
            memory = torch.einsum(
                "bc,clth->blth", basis, self.controls,
            )
            if self.residual_scale != 1.0:
                center = self.controls.mean(0, keepdim=True)
                memory = center + self.residual_scale * (memory - center)
        return memory

    def metadata(self):
        return {
            "num_tasks": self.num_tasks, "degree": self.degree,
            "control_count": int(self.controls.shape[0]),
            "num_reader_layers": self.num_layers,
            "prompt_length": self.prompt_length, "hidden_size": self.hidden_size,
            "residual_scale": self.residual_scale,
        }


def completion_logits(model, prompt_ids, prompt_mask, completion_ids,
                      completion_mask, code, positions):
    """Dispatch to token insertion or the true hidden-residual channel."""
    if _ACTIVE_HIDDEN_RESIDUAL is None:
        return _token_insertion_completion_logits(
            model, prompt_ids, prompt_mask, completion_ids,
            completion_mask, code, positions,
        )
    alpha = _ACTIVE_HIDDEN_RESIDUAL.layer_alphas()
    zero_gate = None
    if alpha is not None:
        alpha = alpha.to(code.device).unsqueeze(0).expand(code.shape[0], -1)
    if _ACTIVE_HIDDEN_RESIDUAL.current_slow_anchor:
        request_gate = code[..., -1].float().reshape(code.shape[0], -1).mean(-1)
        zero_gate = request_gate.eq(0)
        if bool(zero_gate.all()):
            return plain_completion_logits(
                model, prompt_ids, prompt_mask, completion_ids, completion_mask,
            )
        alpha = (
            request_gate.unsqueeze(-1) * alpha
            if alpha is not None else request_gate
        )
    # Never expose the transport scalar to the actual memory K/V projections.
    memory = torch.cat((code[..., :-1], torch.zeros_like(code[..., -1:])), -1)
    _ACTIVE_HIDDEN_RESIDUAL.reader.begin(memory, alpha=alpha)
    try:
        logits = plain_completion_logits(
            model, prompt_ids, prompt_mask, completion_ids, completion_mask,
        )
    finally:
        _ACTIVE_HIDDEN_RESIDUAL.reader.end()
    if zero_gate is not None and bool(zero_gate.any()):
        # Fused preference batches can mix the exact Slow anchor with active
        # memories. Recompute only the zero-gated rows on the literal plain
        # path so their function is bitwise the same as Slow-only rather than
        # merely mathematically equal after multiplying a residual by zero.
        plain = plain_completion_logits(
            model, prompt_ids[zero_gate], prompt_mask[zero_gate],
            completion_ids[zero_gate], completion_mask[zero_gate],
        )
        logits = logits.clone()
        logits[zero_gate] = plain
    return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--buffer", type=Path)
    parser.add_argument(
        "--replay-mode",
        choices=("per_task_one", "sample_fixed", "full_buffer"),
        default="per_task_one",
        help=(
            "per_task_one draws one global row per old task each step; "
            "sample_fixed draws a stratified global subset each step; "
            "full_buffer evaluates every replay-buffer row each step"
        ),
    )
    parser.add_argument(
        "--replay-sample-size", type=int, default=0,
        help=(
            "Global replay subset size used by --replay-mode sample_fixed. "
            "Rows are allocated proportionally across old tasks and traversed "
            "with independent shuffled cycles."
        ),
    )
    parser.add_argument("--previous-prompt", type=Path)
    parser.add_argument(
        "--fresh-fast-controls", action="store_true",
        help=(
            "At stage > 0, initialize every simplex control from the base "
            "model token embeddings instead of loading a previous prompt."
        ),
    )
    parser.add_argument(
        "--preserve-teacher-model",
        help="Frozen previous-stage slow checkpoint used by preserve FKL.",
    )
    parser.add_argument(
        "--fast-only", action="store_true",
        help="Freeze the supplied SFT+Replay checkpoint and optimize only Fast.",
    )
    parser.add_argument(
        "--acquire-only-current-apex", action="store_true",
        help=(
            "Run only the current-task acquisition phase.  The Slow backbone "
            "and the newly introduced task vertex are jointly optimized on "
            "current-task SFT data; replay is deliberately excluded and all "
            "inherited old-face controls are frozen."
        ),
    )
    parser.add_argument(
        "--load-same-stage-prompt", action="store_true",
        help=(
            "Load --previous-prompt as an already expanded prompt with exactly "
            "stage+1 task coordinates instead of recursively expanding it."
        ),
    )
    parser.add_argument(
        "--train-all-fast-controls", action="store_true",
        help="Allow inherited simplex controls to realign to the new frozen Slow.",
    )
    parser.add_argument(
        "--freeze-all-vertices", action="store_true",
        help=(
            "During Fast surface completion, freeze every pure simplex vertex "
            "and optimize only edge/interior controls."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--global-batch", type=int, default=128)
    parser.add_argument(
        "--crossfit-dual-probe-size", type=int, default=0,
        help=(
            "If positive, split both the fixed current pool and replay pool "
            "into a rotating held-out fold of this global size. The complement "
            "is used for primal backward, while the held-out rows update only "
            "the apex dual and never enter backward."
        ),
    )
    parser.add_argument(
        "--heldout-current-dual-probe-size", type=int, default=0,
        help=(
            "Reserve one fixed, deterministic current-task probe pool of this "
            "global size. Each primal step samples a fresh current batch from "
            "the remaining complete training distribution; probe rows update "
            "only the apex dual and never enter backward."
        ),
    )
    parser.add_argument(
        "--heldout-replay-dual-probe-size", type=int, default=0,
        help=(
            "Reserve a fixed replay probe pool for the old-task apex dual. "
            "The remaining replay rows form the primal replay pool."
        ),
    )
    parser.add_argument(
        "--current-batch-match-replay", action="store_true",
        help=(
            "Use the exact logical replay-buffer cardinality as the global "
            "current-task batch size and shard it unevenly across ranks when "
            "necessary."
        ),
    )
    parser.add_argument(
        "--current-with-replacement", action="store_true",
        help=(
            "Draw each logical current-task batch independently with replacement "
            "from the complete eligible current-task training set."
        ),
    )
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument(
        "--hidden-residual", action="store_true",
        help="Use h'=h+aR_phi(h,lambda), not sequence-level soft tokens.",
    )
    parser.add_argument("--reader-scale-init", type=float, default=0.01)
    parser.add_argument(
        "--reader-scale-lr", type=float,
        help="Optional separate learning rate for reader.layer_scales.",
    )
    parser.add_argument("--reader-o-lora-rank", type=int, default=8)
    parser.add_argument("--layer-specific-controls", action="store_true")
    parser.add_argument("--trainable-layer-alpha", action="store_true")
    parser.add_argument(
        "--freeze-memory-reader", action="store_true",
        help=(
            "Freeze the shared hidden-memory reader (including its scales and "
            "W_o LoRA) and optimize only Bezier memory controls. This keeps "
            "functionally rebased task vertices fixed while fitting interior "
            "controls."
        ),
    )
    parser.add_argument(
        "--current-slow-anchor", action="store_true",
        help=(
            "With --hidden-residual, freeze the newest-task pure vertex at "
            "zero and gate every memory layer by 1-lambda_current.  The newest "
            "task vertex is then exactly the unconditioned Slow policy."
        ),
    )
    parser.add_argument("--bezier-degree", type=int, default=3)
    parser.add_argument("--slow-lr", type=float, default=1e-5)
    parser.add_argument("--fast-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--lr-scheduler-type", choices=("linear", "cosine", "constant"),
        default="linear",
    )
    parser.add_argument("--stch-mu", type=float, default=0.1)
    parser.add_argument(
        "--relative-softplus-tau", type=float, default=0.0,
        help=(
            "If positive, replace each per-example task loss ell_fast by "
            "tau*softplus((ell_fast-ell_slow)/tau), where ell_slow is the "
            "frozen unconditioned Slow loss on the same example."
        ),
    )
    parser.add_argument("--batch-regret-margin", type=float, default=0.0)
    parser.add_argument("--batch-regret-tau", type=float, default=0.01)
    parser.add_argument("--batch-regret-weight", type=float, default=0.0)
    parser.add_argument("--taskwise-regret-margin", type=float, default=0.0)
    parser.add_argument("--taskwise-regret-tau", type=float, default=0.01)
    parser.add_argument("--taskwise-dual-lr", type=float, default=0.0)
    parser.add_argument("--taskwise-dual-init", type=float, default=1.0)
    parser.add_argument("--taskwise-dual-max", type=float, default=10.0)
    parser.add_argument(
        "--apex-specific-taskwise-dual", action="store_true",
        help=(
            "Enforce only Delta_t(e_t) <= -m_t, update each dual from its "
            "matching exact vertex, and prevent non-vertex preferences from "
            "updating pure Bezier vertex controls."
        ),
    )
    parser.add_argument(
        "--taskwise-dual-ema-decay", type=float, default=0.9,
        help="EMA decay for per-task mean constraint violations used by dual ascent",
    )
    parser.add_argument(
        "--disable-stch-normalization", action="store_true",
        help="Apply STCH directly to raw task objectives without min/max scaling.",
    )
    parser.add_argument(
        "--scalarization", choices=("stch", "weighted_sum"), default="stch",
        help=(
            "Preference scalarization used for Fast optimization. weighted_sum "
            "uses sum_t lambda_t L_t directly and never normalizes task losses."
        ),
    )
    parser.add_argument("--range-floor", type=float, default=0.05)
    parser.add_argument("--preserve-top-k", type=int, default=64)
    parser.add_argument(
        "--preserve-objective", choices=("sft", "topk_fkl"),
        default="topk_fkl",
        help=(
            "Replay supervision for Fast: gold-answer SFT or old-checkpoint "
            "top-k+OTHER teacher-forcing forward KL."
        ),
    )
    parser.add_argument(
        "--joint-projection", action="store_true",
        help="One-phase simultaneous slow conflict-projection and fast STCH.",
    )
    parser.add_argument("--fast-calibration-steps", type=int, default=10)
    parser.add_argument(
        "--current-apex-warmup-steps", type=int, default=0,
        help=(
            "Before fast-only simplex training, freeze Slow and fit only the "
            "current-task simplex vertex with current-task SFT for this many "
            "optimizer steps. The subsequent --epochs pass is unchanged."
        ),
    )
    parser.add_argument("--fast-residual-scale", type=float, default=1.0)
    parser.add_argument("--condition-chunk", type=int, default=4)
    parser.add_argument(
        "--condition-token-budget", type=int, default=0,
        help=(
            "If positive, dynamically choose a row chunk up to condition-chunk "
            "so preference_count*rows*max_sequence_tokens stays within this "
            "budget."
        ),
    )
    parser.add_argument("--preference-count", type=int, default=4)
    parser.add_argument(
        "--preference-dirichlet-alpha", type=float, default=1.0,
        help="Symmetric Dirichlet concentration for random preferences.",
    )
    parser.add_argument(
        "--include-one-cyclic-vertex", action="store_true",
        help=(
            "Make the first preference a simplex unit vector, cycling across "
            "task vertices by optimizer step; the remaining points are random."
        ),
    )
    parser.add_argument(
        "--recursive-stratified-preferences", action="store_true",
        help=(
            "Use recursive-simplex coverage: one new-task apex, one old face, "
            "two old/new edges, and the remaining points from the requested "
            "symmetric Dirichlet distribution."
        ),
    )
    parser.add_argument(
        "--preference-chunk", type=int, default=1,
        help="Number of preferences fused into each conditional forward/backward.",
    )
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument(
        "--replay-overlength", choices=("drop", "head_tail"), default="drop",
        help=(
            "How to handle replay prompts longer than max-prompt-length. "
            "head_tail retains the task/instruction prefix and the problem/"
            "assistant suffix so the cumulative buffer cardinality is exact."
        ),
    )
    parser.add_argument("--max-completion-length", type=int)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--length-bucket-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--init-text",
        default="Answer the question accurately and follow the required output format.",
    )
    return parser.parse_args()


def sample_preferences(num_tasks: int, step: int, seed: int, device: torch.device):
    """Uniform barycenter followed by three reproducible Dirichlet(1) draws."""
    uniform = torch.full((num_tasks,), 1.0 / num_tasks)
    if num_tasks == 1:
        return uniform.unsqueeze(0).to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 104729 * step + 1000003 * num_tasks)
    values = torch.rand(3, num_tasks, generator=generator).clamp_min_(1e-7)
    random_preferences = -values.log()
    random_preferences /= random_preferences.sum(-1, keepdim=True)
    return torch.cat((uniform.unsqueeze(0), random_preferences), dim=0).to(device)


def sample_random_preferences(
    num_tasks: int, step: int, seed: int, device: torch.device,
    count: int = 4, alpha: float = 1.0, include_one_cyclic_vertex: bool = False,
) -> torch.Tensor:
    """Reproducible interior symmetric-Dirichlet preferences."""
    if num_tasks == 1:
        return torch.ones(1, 1, device=device)
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 130363 * step + 2000003 * num_tasks)
    random_count = count - int(include_one_cyclic_vertex)
    if random_count < 0:
        raise ValueError("preference count must be at least one")
    concentration = torch.full((random_count, num_tasks), float(alpha))
    preferences = torch._standard_gamma(concentration, generator=generator)
    preferences.clamp_min_(1e-12)
    preferences /= preferences.sum(-1, keepdim=True)
    if include_one_cyclic_vertex:
        vertex = torch.zeros(1, num_tasks)
        vertex[0, (step - 1) % num_tasks] = 1.0
        preferences = torch.cat((vertex, preferences), dim=0)
    return preferences.to(device)


def sample_recursive_preferences(
    num_tasks: int, step: int, seed: int, device: torch.device,
    count: int = 8, alpha: float = 0.5,
) -> torch.Tensor:
    """Stratified coverage for a recursively expanded task simplex.

    The old-face point is intentionally retained as a zero-gradient boundary
    diagnostic after that face is frozen.  The two edge points ensure the
    newly introduced degree-1/2 Bernstein strata receive signal even when a
    high-dimensional Dirichlet draw is concentrated in the interior.
    """
    if num_tasks == 1:
        return torch.ones(count, 1, device=device)
    if count < 4:
        raise ValueError("recursive stratified sampling requires count >= 4")
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 15485863 * step + 32452843 * num_tasks)

    new_apex = torch.zeros(1, num_tasks)
    new_apex[0, -1] = 1.0

    old_concentration = torch.full((1, num_tasks - 1), float(alpha))
    old_face = torch._standard_gamma(old_concentration, generator=generator)
    old_face.clamp_min_(1e-12)
    old_face /= old_face.sum(-1, keepdim=True)
    old_face = torch.cat((old_face, torch.zeros(1, 1)), dim=-1)

    edge_rows = []
    for edge_index, new_mass in enumerate((0.25, 0.75)):
        row = torch.zeros(num_tasks)
        old_vertex = (step - 1 + edge_index) % (num_tasks - 1)
        row[old_vertex] = 1.0 - new_mass
        row[-1] = new_mass
        edge_rows.append(row)
    edges = torch.stack(edge_rows)

    interior_count = count - 4
    if interior_count:
        concentration = torch.full((interior_count, num_tasks), float(alpha))
        interior = torch._standard_gamma(concentration, generator=generator)
        interior.clamp_min_(1e-12)
        interior /= interior.sum(-1, keepdim=True)
    else:
        interior = torch.empty(0, num_tasks)
    return torch.cat((new_apex, old_face, edges, interior), dim=0).to(device)


def sample_joint_preferences(
    num_tasks: int, step: int, seed: int, device: torch.device,
) -> torch.Tensor:
    """Two full-simplex draws plus two old-face draws.

    Old-face preferences set the current (last) task weight to exactly zero and
    distribute unit mass randomly over all previously seen tasks.
    """
    if num_tasks < 2:
        return torch.ones(4, 1, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 32452843 * step + 49979687 * num_tasks)
    full_u = torch.rand(2, num_tasks, generator=generator).clamp_min_(1e-7)
    full = -full_u.log()
    full /= full.sum(-1, keepdim=True)
    old_u = torch.rand(2, num_tasks - 1, generator=generator).clamp_min_(1e-7)
    old = -old_u.log()
    old /= old.sum(-1, keepdim=True)
    old_face = torch.cat((old, torch.zeros(2, 1)), dim=-1)
    return torch.cat((full, old_face), dim=0).to(device)


def build_replay_schedule(
    pools: dict[int, list[dict]], total_steps: int, seed: int,
    *, mode: str = "per_task_one", rank: int = 0, world: int = 1,
    sample_size: int = 0,
) -> list[list[dict]]:
    """One global row per old task and step, sampled without replacement.

    Every rank intentionally receives the same rows. DDP gradient summation
    below divides their local replay contribution by the global count, so this
    is one logical global example rather than ``world_size`` examples. This
    avoids autograd-aware variable-size cross-rank gathers while preserving the
    exact global-one-sample objective.
    """
    if mode == "full_buffer":
        # The full logical buffer is deterministically sharded across ranks.
        # distributed_task_means() later all-reduces sums and counts, so this
        # is exactly the global full-buffer objective without 4x duplication.
        all_rows = [
            row for task_id in sorted(pools) for row in pools[task_id]
        ]
        local_rows = all_rows[rank::world]
        if not local_rows:
            raise RuntimeError(
                f"empty local full-buffer shard rank={rank} world={world}"
            )
        return [local_rows for _ in range(total_steps)]
    if mode == "sample_fixed":
        total_rows = sum(len(pool) for pool in pools.values())
        if sample_size <= 0:
            raise ValueError("sample_fixed requires --replay-sample-size > 0")
        if sample_size > total_rows:
            raise ValueError(
                f"replay sample size {sample_size} exceeds buffer size {total_rows}"
            )

        # Preserve the buffer's task allocation in every logical global batch.
        # Largest-remainder rounding makes the per-task counts sum exactly to
        # sample_size, while the lower bound keeps every old objective present.
        task_ids = sorted(pools)
        if sample_size < len(task_ids):
            raise ValueError(
                f"replay sample size {sample_size} is smaller than "
                f"the number of old tasks {len(task_ids)}"
            )
        exact = {
            task_id: sample_size * len(pools[task_id]) / total_rows
            for task_id in task_ids
        }
        counts = {task_id: max(1, int(exact[task_id])) for task_id in task_ids}
        while sum(counts.values()) < sample_size:
            task_id = max(task_ids, key=lambda t: (exact[t] - counts[t], -t))
            counts[task_id] += 1
        while sum(counts.values()) > sample_size:
            candidates = [t for t in task_ids if counts[t] > 1]
            task_id = min(candidates, key=lambda t: (exact[t] - counts[t], t))
            counts[task_id] -= 1

        streams: dict[int, list[dict]] = {}
        for task_id in task_ids:
            required = total_steps * counts[task_id]
            stream: list[dict] = []
            cycle = 0
            while len(stream) < required:
                indices = list(range(len(pools[task_id])))
                random.Random(
                    seed + task_id * 9176 + cycle * 104729
                ).shuffle(indices)
                stream.extend(pools[task_id][index] for index in indices)
                cycle += 1
            streams[task_id] = stream[:required]

        schedule: list[list[dict]] = []
        offsets = {task_id: 0 for task_id in task_ids}
        for _ in range(total_steps):
            global_rows: list[dict] = []
            for task_id in task_ids:
                begin = offsets[task_id]
                end = begin + counts[task_id]
                global_rows.extend(streams[task_id][begin:end])
                offsets[task_id] = end
            schedule.append(global_rows[rank::world])
        return schedule
    if mode != "per_task_one":
        raise ValueError(f"unknown replay mode: {mode}")

    schedules: dict[int, list[dict]] = {}
    for task_id, pool in sorted(pools.items()):
        if not pool:
            raise RuntimeError(f"empty replay pool for task id {task_id}")
        sequence: list[dict] = []
        cycle = 0
        while len(sequence) < total_steps:
            indices = list(range(len(pool)))
            random.Random(seed + task_id * 9176 + cycle * 104729).shuffle(indices)
            sequence.extend(pool[index] for index in indices)
            cycle += 1
        schedules[task_id] = sequence[:total_steps]
    return [
        [schedules[task_id][step] for task_id in sorted(schedules)]
        for step in range(total_steps)
    ]


def batch_tensors(rows, tokenizer, device):
    prompt_ids, prompt_mask, _ = pad_id_rows(
        [row["prompt_ids"] for row in rows], tokenizer.pad_token_id, device,
    )
    width = prompt_ids.shape[1]
    positions = torch.tensor([
        # Insert after left padding but before the first token of the complete
        # rendered chat (system/user/assistant-control tokens included).
        width - len(row["prompt_ids"])
        for row in rows
    ], dtype=torch.long, device=device)
    target_ids, target_mask, _ = pad_id_rows(
        [row["answer_ids"] for row in rows], tokenizer.pad_token_id, device,
        padding_side="right",
    )
    task_ids = torch.tensor(
        [row["task_id"] for row in rows], dtype=torch.long, device=device,
    )
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "positions": positions,
        "target_ids": target_ids,
        "target_mask": target_mask,
        "task_ids": task_ids,
    }


def fast_condition_tensors(
    current_rows, replay_rows, tokenizer, device, preserve_objective,
):
    """Build one length-sorted condition batch when both targets are SFT."""
    if replay_rows and preserve_objective == "sft":
        merged = sorted(
            [*current_rows, *replay_rows],
            key=lambda row: int(row["length"]),
        )
        return batch_tensors(merged, tokenizer, device), None, True
    return (
        batch_tensors(current_rows, tokenizer, device),
        batch_tensors(replay_rows, tokenizer, device) if replay_rows else None,
        False,
    )


def compact_tensor_rows(tensors, start, end):
    """Slice a row block and remove padding inherited from the outer batch."""
    prompt_width = int(
        tensors["prompt_mask"][start:end].sum(-1).max().item()
    )
    target_width = int(
        tensors["target_mask"][start:end].sum(-1).max().item()
    )
    prompt_trim = tensors["prompt_ids"].shape[1] - prompt_width
    block = {
        "prompt_ids": tensors["prompt_ids"][start:end, -prompt_width:],
        "prompt_mask": tensors["prompt_mask"][start:end, -prompt_width:],
        "positions": tensors["positions"][start:end] - prompt_trim,
        "target_ids": tensors["target_ids"][start:end, :target_width],
        "target_mask": tensors["target_mask"][start:end, :target_width],
        "task_ids": tensors["task_ids"][start:end],
    }
    if "slow_reference_loss" in tensors:
        block["slow_reference_loss"] = tensors["slow_reference_loss"][start:end]
    return block


@torch.no_grad()
def attach_slow_reference_losses(model, tensors, condition_chunk):
    """Cache the plain frozen-Slow loss for each row in a logical batch."""
    if tensors is None:
        return
    was_training = model.training
    model.eval()
    losses = []
    row_count = len(tensors["task_ids"])
    for start in range(0, row_count, condition_chunk):
        end = min(start + condition_chunk, row_count)
        block = compact_tensor_rows(tensors, start, end)
        logits = plain_completion_logits(
            model, block["prompt_ids"], block["prompt_mask"],
            block["target_ids"], block["target_mask"],
        )
        losses.append(sft_nll_per_example(
            logits, block["target_ids"], block["target_mask"],
        ).detach())
    tensors["slow_reference_loss"] = torch.cat(losses, dim=0)
    model.train(was_training)


def relative_softplus_losses(losses, slow_losses, tau):
    """Smooth relative-regret loss in raw per-token NLL units."""
    if tau <= 0:
        return losses
    return tau * torch.nn.functional.softplus(
        (losses - slow_losses.detach()) / tau
    )


def adaptive_row_ranges(
    tensors, max_chunk, preference_count, prompt_length, token_budget,
):
    """Yield the largest safe power-of-two row blocks by actual token length."""
    row_count = len(tensors["task_ids"])
    lengths = (
        tensors["prompt_mask"].sum(-1)
        + tensors["target_mask"].sum(-1)
        + int(prompt_length)
    )
    start = 0
    while start < row_count:
        upper = min(max_chunk, row_count - start)
        # Build the fallback ladder from the requested maximum rather than
        # silently capping it at four.  The token budget remains the final
        # guard for long sequences (e.g. 8 -> 4 -> 2 -> 1).
        candidates = []
        value = upper
        while value >= 1:
            candidates.append(value)
            if value == 1:
                break
            value = max(1, value // 2)
        if upper not in candidates:
            candidates.insert(0, upper)
        chosen = 1
        for rows in sorted(set(candidates), reverse=True):
            maximum = int(lengths[start:start + rows].max().item())
            cost = preference_count * rows * maximum
            if token_budget <= 0 or cost <= token_budget or rows == 1:
                chosen = rows
                break
        yield start, start + chosen
        start += chosen


def distributed_task_means(local_sums: torch.Tensor, local_counts: torch.Tensor):
    global_sums = local_sums.detach().clone()
    global_counts = local_counts.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(global_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    if torch.any(global_counts <= 0):
        missing = torch.nonzero(global_counts <= 0).flatten().tolist()
        raise RuntimeError(f"task objectives missing from distributed batch: {missing}")
    detached_means = global_sums / global_counts
    # Exact global value with a local differentiable contribution.  Summing
    # (not averaging) gradients across ranks later recovers the global mean.
    local_part = local_sums / global_counts
    means_with_local_grad = local_part + (detached_means - local_part.detach())
    return means_with_local_grad, detached_means, global_counts


def stch_scalar(
    task_losses: torch.Tensor,
    preference: torch.Tensor,
    normalization: dict[str, torch.Tensor],
    mu: float,
    normalize: bool,
):
    if normalize and normalization["initialized"]:
        running_min = normalization["running_min"].to(task_losses.device)
        running_max = normalization["running_max"].to(task_losses.device)
        ideal = 0.9 * running_min
        scales = (running_max - ideal).clamp_min(normalization["range_floor"])
        values = (task_losses - ideal) / scales
    else:
        values = task_losses
    weighted = preference * values
    return mu * torch.logsumexp(weighted / mu, dim=0), values


def update_running_normalization(
    normalization: dict[str, torch.Tensor], detached_losses: torch.Tensor,
) -> None:
    """Update detached cumulative min/max without an extra model forward."""
    values = detached_losses.detach().float().cpu()
    if values.ndim == 1:
        values = values.unsqueeze(0)
    batch_min = values.min(0).values
    batch_max = values.max(0).values
    if not normalization["initialized"]:
        normalization["running_min"] = batch_min
        normalization["running_max"] = batch_max
        normalization["initialized"] = True
    else:
        normalization["running_min"] = torch.minimum(
            normalization["running_min"], batch_min,
        )
        normalization["running_max"] = torch.maximum(
            normalization["running_max"], batch_max,
        )


def allreduce_grads(parameters: list[torch.nn.Parameter], *, average: bool):
    present = [parameter for parameter in parameters if parameter.grad is not None]
    if not present or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    flat = torch.cat([parameter.grad.reshape(-1) for parameter in present])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    if average:
        flat.div_(dist.get_world_size())
    offset = 0
    for parameter in present:
        count = parameter.numel()
        parameter.grad.copy_(flat[offset:offset + count].view_as(parameter))
        offset += count


def allreduce_grad_list(grads: list[torch.Tensor]) -> None:
    """Sum a detached gradient list across ranks with one collective."""
    if not grads or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    flat = torch.cat([grad.reshape(-1) for grad in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for grad in grads:
        count = grad.numel()
        grad.copy_(flat[offset:offset + count].view_as(grad))
        offset += count


def accumulate_grad_list(
    accumulator: list[torch.Tensor] | None,
    grads: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor]:
    if accumulator is None:
        accumulator = [torch.zeros_like(parameter) for parameter in parameters]
    for total, grad in zip(accumulator, grads):
        if grad is not None:
            total.add_(grad.detach())
    return accumulator


@torch.no_grad()
def preserve_teacher_targets(
    teacher,
    old_prompt_model,
    tensors,
    top_k,
    condition_chunk,
):
    """Score each old-task replay row once with the frozen teacher.

    Legacy joint training conditions the teacher on the corresponding old
    simplex vertex. Fast-only training deliberately passes ``None`` and uses
    the previous stage's plain SFT+Replay behavior as the functional target.
    """
    task_ids = tensors["task_ids"]
    if old_prompt_model is not None:
        old_task_count = old_prompt_model.num_tasks
        preferences = torch.nn.functional.one_hot(
            task_ids, num_classes=old_task_count,
        ).to(dtype=torch.float32)
    top_indices, top_logps, top_masses = [], [], []
    for start in range(0, len(task_ids), condition_chunk):
        end = min(start + condition_chunk, len(task_ids))
        if old_prompt_model is None:
            logits = plain_completion_logits(
                teacher,
                tensors["prompt_ids"][start:end],
                tensors["prompt_mask"][start:end],
                tensors["target_ids"][start:end],
                tensors["target_mask"][start:end],
            )
        else:
            code = old_prompt_model(preferences[start:end])
            logits = completion_logits(
                teacher,
                tensors["prompt_ids"][start:end], tensors["prompt_mask"][start:end],
                tensors["target_ids"][start:end], tensors["target_mask"][start:end],
                code, tensors["positions"][start:end],
            )
        top_idx, top_logp, top_mass = teacher_topk_tail_targets(
            logits, top_k=top_k,
        )
        top_indices.append(top_idx)
        top_logps.append(top_logp)
        top_masses.append(top_mass)
    return (
        torch.cat(top_indices), torch.cat(top_logps), torch.cat(top_masses),
    )


def preference_forward(
    model,
    prompt_model,
    current_tensors,
    replay_tensors,
    preserve_targets,
    preference,
    num_tasks,
    condition_chunk,
):
    local_sums = torch.zeros(num_tasks, device=preference.device)
    local_counts = torch.zeros(num_tasks, device=preference.device)
    row_count = len(current_tensors["task_ids"])
    for start in range(0, row_count, condition_chunk):
        end = min(start + condition_chunk, row_count)
        count = end - start
        preferences = preference.unsqueeze(0).expand(count, -1)
        code = prompt_model(preferences)
        logits = completion_logits(
            model,
            current_tensors["prompt_ids"][start:end],
            current_tensors["prompt_mask"][start:end],
            current_tensors["target_ids"][start:end],
            current_tensors["target_mask"][start:end],
            code, current_tensors["positions"][start:end],
        )
        losses = sft_nll_per_example(
            logits, current_tensors["target_ids"][start:end],
            current_tensors["target_mask"][start:end],
        )
        task_ids = current_tensors["task_ids"][start:end]
        local_sums = local_sums.scatter_add(0, task_ids, losses)
        local_counts.scatter_add_(0, task_ids, torch.ones_like(losses.detach()))

    if replay_tensors is not None:
        if preserve_targets is not None:
            top_idx, top_logp, top_mass = preserve_targets
        replay_count = len(replay_tensors["task_ids"])
        for start in range(0, replay_count, condition_chunk):
            end = min(start + condition_chunk, replay_count)
            count = end - start
            preferences = preference.unsqueeze(0).expand(count, -1)
            code = prompt_model(preferences)
            logits = completion_logits(
                model,
                replay_tensors["prompt_ids"][start:end],
                replay_tensors["prompt_mask"][start:end],
                replay_tensors["target_ids"][start:end],
                replay_tensors["target_mask"][start:end],
                code, replay_tensors["positions"][start:end],
            )
            if preserve_targets is None:
                losses = sft_nll_per_example(
                    logits,
                    replay_tensors["target_ids"][start:end],
                    replay_tensors["target_mask"][start:end],
                )
            else:
                token_kl, _, _ = topk_tail_forward_kl_from_targets(
                    top_idx[start:end], top_logp[start:end],
                    top_mass[start:end], logits,
                )
                mask = replay_tensors["target_mask"][start:end]
                losses = (token_kl * mask).sum(-1) / mask.sum(-1).clamp_min(1)
            task_ids = replay_tensors["task_ids"][start:end]
            local_sums = local_sums.scatter_add(0, task_ids, losses)
            local_counts.scatter_add_(
                0, task_ids, torch.ones_like(losses.detach()),
            )
    return distributed_task_means(local_sums, local_counts)


def preference_forward_batch(
    model,
    prompt_model,
    current_tensors,
    replay_tensors,
    preserve_targets,
    preferences,
    num_tasks,
    condition_chunk,
    condition_token_budget=0,
    relative_softplus_tau=0.0,
):
    """Evaluate multiple preferences in fused preference-by-row batches.

    ``condition_chunk`` counts rows per preference.  A chunk with ``P``
    preferences therefore sends ``P * condition_chunk`` sequences through a
    single model call.  This preserves the exact Cartesian-product objective
    while increasing GPU utilization.
    """
    preference_count = len(preferences)
    local_sums = torch.zeros(
        preference_count, num_tasks, device=preferences.device,
    )
    local_counts = torch.zeros_like(local_sums)
    relative_delta_sums = torch.zeros_like(local_sums)
    relative_worse_sums = torch.zeros_like(local_sums)
    relative_weight_sums = torch.zeros_like(local_sums)

    def score_rows(tensors, targets):
        nonlocal local_sums, local_counts
        nonlocal relative_delta_sums, relative_worse_sums, relative_weight_sums
        for start, end in adaptive_row_ranges(
            tensors, condition_chunk, preference_count,
            prompt_model.controls.shape[-2], condition_token_budget,
        ):
            block = compact_tensor_rows(tensors, start, end)
            rows = end - start
            flat_preferences = preferences[:, None, :].expand(
                preference_count, rows, num_tasks,
            ).reshape(preference_count * rows, num_tasks)
            code = prompt_model(flat_preferences)

            def expand(name):
                value = block[name]
                return value.unsqueeze(0).expand(
                    preference_count, *value.shape,
                ).reshape(preference_count * rows, *value.shape[1:])

            logits = completion_logits(
                model, expand("prompt_ids"), expand("prompt_mask"),
                expand("target_ids"), expand("target_mask"), code,
                expand("positions"),
            )
            target_ids = expand("target_ids")
            target_mask = expand("target_mask")
            if targets is None:
                losses = sft_nll_per_example(
                    logits, target_ids, target_mask,
                ).reshape(preference_count, rows)
            else:
                top_idx, top_logp, top_mass = targets

                def expand_target(value):
                    value = value[start:end, :block["target_ids"].shape[1]]
                    return value.unsqueeze(0).expand(
                        preference_count, *value.shape,
                    ).reshape(preference_count * rows, *value.shape[1:])

                token_kl, _, _ = topk_tail_forward_kl_from_targets(
                    expand_target(top_idx), expand_target(top_logp),
                    expand_target(top_mass), logits,
                )
                losses = (
                    (token_kl * target_mask).sum(-1)
                    / target_mask.sum(-1).clamp_min(1)
                ).reshape(preference_count, rows)
            task_ids = block["task_ids"].unsqueeze(0).expand(
                preference_count, rows,
            )
            if relative_softplus_tau > 0:
                slow_losses = block["slow_reference_loss"].unsqueeze(0).expand(
                    preference_count, rows,
                )
                deltas = losses.detach() - slow_losses
                relative_delta_sums.scatter_add_(1, task_ids, deltas)
                relative_worse_sums.scatter_add_(
                    1, task_ids, (deltas > 0).to(losses.dtype),
                )
                relative_weight_sums.scatter_add_(
                    1, task_ids,
                    torch.sigmoid(deltas / relative_softplus_tau),
                )
                losses = relative_softplus_losses(
                    losses, slow_losses, relative_softplus_tau,
                )
            local_sums.scatter_add_(1, task_ids, losses)
            local_counts.scatter_add_(
                1, task_ids, torch.ones_like(losses.detach()),
            )

    score_rows(current_tensors, None)
    if replay_tensors is not None:
        score_rows(replay_tensors, preserve_targets)
    means, detached_means, global_counts = distributed_task_means(
        local_sums, local_counts,
    )
    diagnostics = None
    if relative_softplus_tau > 0:
        for values in (
            relative_delta_sums, relative_worse_sums, relative_weight_sums,
        ):
            if dist.is_initialized():
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
        denominator = global_counts.clamp_min(1)
        diagnostics = {
            "relative_delta_mean_by_preference": (
                relative_delta_sums / denominator
            ).detach(),
            "relative_worse_fraction_by_preference": (
                relative_worse_sums / denominator
            ).detach(),
            "relative_gradient_weight_mean_by_preference": (
                relative_weight_sums / denominator
            ).detach(),
        }
    return means, detached_means, global_counts, diagnostics


def preference_backward_streaming(
    model,
    prompt_model,
    current_tensors,
    replay_tensors,
    preserve_targets,
    preferences,
    task_coefficients,
    global_counts,
    condition_chunk,
    condition_token_budget=0,
    relative_softplus_tau=0.0,
    raw_task_coefficients=None,
    isolate_vertex_gradients=False,
):
    """Backpropagate an exact fused multi-preference objective by row block.

    ``task_coefficients[p, t]`` is the derivative of the scalarized loss with
    respect to the global mean loss for preference ``p`` and task ``t``.
    Dividing it by the corresponding global row count makes each local block
    an exact contribution to that global mean.  Backward is performed as soon
    as a block has been scored, so activation memory is bounded by
    ``len(preferences) * condition_chunk`` rather than the whole batch.
    """
    preference_count, num_tasks = preferences.shape

    def score_rows(tensors, targets):
        for start, end in adaptive_row_ranges(
            tensors, condition_chunk, preference_count,
            prompt_model.controls.shape[-2], condition_token_budget,
        ):
            block = compact_tensor_rows(tensors, start, end)
            rows = end - start
            flat_preferences = preferences[:, None, :].expand(
                preference_count, rows, num_tasks,
            ).reshape(preference_count * rows, num_tasks)
            code = prompt_model(
                flat_preferences,
                isolate_vertex_gradients=isolate_vertex_gradients,
            )

            def expand(name):
                value = block[name]
                return value.unsqueeze(0).expand(
                    preference_count, *value.shape,
                ).reshape(preference_count * rows, *value.shape[1:])

            logits = completion_logits(
                model, expand("prompt_ids"), expand("prompt_mask"),
                expand("target_ids"), expand("target_mask"), code,
                expand("positions"),
            )
            target_ids = expand("target_ids")
            target_mask = expand("target_mask")
            if targets is None:
                losses = sft_nll_per_example(
                    logits, target_ids, target_mask,
                ).reshape(preference_count, rows)
            else:
                top_idx, top_logp, top_mass = targets

                def expand_target(value):
                    value = value[start:end, :block["target_ids"].shape[1]]
                    return value.unsqueeze(0).expand(
                        preference_count, *value.shape,
                    ).reshape(preference_count * rows, *value.shape[1:])

                token_kl, _, _ = topk_tail_forward_kl_from_targets(
                    expand_target(top_idx), expand_target(top_logp),
                    expand_target(top_mass), logits,
                )
                losses = (
                    (token_kl * target_mask).sum(-1)
                    / target_mask.sum(-1).clamp_min(1)
                ).reshape(preference_count, rows)
            raw_losses = losses
            if relative_softplus_tau > 0:
                slow_losses = block["slow_reference_loss"].unsqueeze(0).expand(
                    preference_count, rows,
                )
                losses = relative_softplus_losses(
                    losses, slow_losses, relative_softplus_tau,
                )
            task_ids = block["task_ids"].unsqueeze(0).expand(
                preference_count, rows,
            )
            row_coefficients = task_coefficients.gather(1, task_ids)
            row_counts = global_counts.gather(1, task_ids).clamp_min(1)
            objective = losses * row_coefficients
            if raw_task_coefficients is not None:
                raw_row_coefficients = raw_task_coefficients.gather(1, task_ids)
                objective = objective + raw_losses * raw_row_coefficients
            ((objective / row_counts).sum()).backward()

    score_rows(current_tensors, None)
    if replay_tensors is not None:
        score_rows(replay_tensors, preserve_targets)


def mask_old_face_gradient(
    prompt_model: SimplexBezierPrompt, old_face_mask: torch.Tensor,
) -> None:
    """Keep the inherited lower-dimensional simplex face bitwise fixed."""
    if prompt_model.controls.grad is not None:
        prompt_model.controls.grad[old_face_mask] = 0


def current_apex_acquire_step(
    model,
    prompt_model,
    current_tensors,
    slow_optimizer,
    fast_optimizer,
    slow_scheduler,
    fast_scheduler,
    old_face_mask,
    args,
):
    """Jointly fit Slow and the new task vertex using current SFT only.

    The exact apex has zero Bernstein support on every inherited old-face
    control.  We additionally mask that face as an invariant check, ensuring
    that acquisition cannot mutate historical Fast memory before rebase.
    """
    slow_parameters = [p for p in model.parameters() if p.requires_grad]
    fast_parameters = list(prompt_model.parameters())
    slow_optimizer.zero_grad(set_to_none=True)
    fast_optimizer.zero_grad(set_to_none=True)
    model.train()
    prompt_model.train()

    task_count = args.stage + 1
    apex = torch.zeros(task_count, device=current_tensors["prompt_ids"].device)
    apex[-1] = 1.0
    local_count = torch.tensor(
        float(len(current_tensors["task_ids"])), device=apex.device,
    )
    global_count = local_count.clone()
    if dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    local_loss_sum = torch.zeros((), device=apex.device)
    row_count = len(current_tensors["task_ids"])
    for start in range(0, row_count, args.condition_chunk):
        end = min(start + args.condition_chunk, row_count)
        count = end - start
        preferences = apex.unsqueeze(0).expand(count, -1)
        code = prompt_model(preferences)
        logits = completion_logits(
            model,
            current_tensors["prompt_ids"][start:end],
            current_tensors["prompt_mask"][start:end],
            current_tensors["target_ids"][start:end],
            current_tensors["target_mask"][start:end],
            code, current_tensors["positions"][start:end],
        )
        losses = sft_nll_per_example(
            logits,
            current_tensors["target_ids"][start:end],
            current_tensors["target_mask"][start:end],
        )
        local_loss_sum = local_loss_sum + losses.detach().sum()
        (losses.sum() / global_count).backward()

    allreduce_grads(slow_parameters, average=False)
    allreduce_grads(fast_parameters, average=False)
    mask_old_face_gradient(prompt_model, old_face_mask)
    slow_grad_norm = torch.nn.utils.clip_grad_norm_(
        slow_parameters, args.max_grad_norm,
    )
    fast_grad_norm = torch.nn.utils.clip_grad_norm_(
        fast_parameters, args.max_grad_norm,
    )
    slow_optimizer.step()
    fast_optimizer.step()
    slow_scheduler.step()
    fast_scheduler.step()

    global_loss_sum = local_loss_sum.clone()
    if dist.is_initialized():
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
    controls = prompt_model.controls.detach().float()
    centered = controls - controls.mean(0, keepdim=True)
    current_apex_sft_nll = float(global_loss_sum / global_count)
    return {
        "phase": "A_current_apex_joint_sft",
        # Keep the precise name and generic aliases so stage-wise log tooling
        # can compare this single-objective Acquire phase with other trainers.
        "loss": current_apex_sft_nll,
        "acquire_nll": current_apex_sft_nll,
        "current_apex_sft_nll": current_apex_sft_nll,
        "slow_grad_norm": float(slow_grad_norm),
        "fast_grad_norm": float(fast_grad_norm),
        "control_norm": float(controls.norm()),
        "control_centered_norm": float(centered.norm()),
        "control_pairwise_rms": float(centered.pow(2).mean().sqrt()),
        "slow_lr": slow_scheduler.get_last_lr()[0],
        "fast_lr": fast_scheduler.get_last_lr()[0],
    }


def fast_only_current_apex_warmup_step(
    model, prompt_model, current_tensors, fast_optimizer, fast_scheduler, args,
):
    """Fit only the new task vertex while keeping the Slow checkpoint frozen."""
    model.requires_grad_(False)
    model.train()
    prompt_model.requires_grad_(True)
    if args.freeze_memory_reader and hasattr(prompt_model, "reader"):
        prompt_model.reader.requires_grad_(False)
        if getattr(prompt_model, "layer_alpha_logits", None) is not None:
            prompt_model.layer_alpha_logits.requires_grad_(False)
    prompt_model.train()
    fast_optimizer.zero_grad(set_to_none=True)

    task_count = args.stage + 1
    apex = torch.zeros(task_count, device=current_tensors["prompt_ids"].device)
    apex[-1] = 1.0
    local_count = torch.tensor(
        float(len(current_tensors["task_ids"])), device=apex.device,
    )
    global_count = local_count.clone()
    if dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)

    local_loss_sum = torch.zeros((), device=apex.device)
    row_count = len(current_tensors["task_ids"])
    for start in range(0, row_count, args.condition_chunk):
        end = min(start + args.condition_chunk, row_count)
        count = end - start
        code = prompt_model(apex.unsqueeze(0).expand(count, -1))
        logits = completion_logits(
            model,
            current_tensors["prompt_ids"][start:end],
            current_tensors["prompt_mask"][start:end],
            current_tensors["target_ids"][start:end],
            current_tensors["target_mask"][start:end],
            code,
            current_tensors["positions"][start:end],
        )
        losses = sft_nll_per_example(
            logits,
            current_tensors["target_ids"][start:end],
            current_tensors["target_mask"][start:end],
        )
        local_loss_sum += losses.detach().sum()
        (losses.sum() / global_count).backward()

    fast_parameters = list(prompt_model.parameters())
    allreduce_grads(fast_parameters, average=False)
    apex_control_mask = prompt_model.multi_indices[:, -1].eq(args.bezier_degree)
    if prompt_model.controls.grad is not None:
        prompt_model.controls.grad[~apex_control_mask] = 0
    fast_grad_norm = torch.nn.utils.clip_grad_norm_(
        fast_parameters, args.max_grad_norm,
    )
    fast_optimizer.step()
    fast_scheduler.step()

    global_loss_sum = local_loss_sum.clone()
    if dist.is_initialized():
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
    controls = prompt_model.controls.detach().float()
    centered = controls - controls.mean(0, keepdim=True)
    loss_value = float(global_loss_sum / global_count)
    return {
        "phase": "A_fast_current_apex_sft_warmup",
        "loss": loss_value,
        "acquire_nll": loss_value,
        "current_apex_sft_nll": loss_value,
        "slow_grad_norm": 0.0,
        "fast_grad_norm": float(fast_grad_norm),
        "control_norm": float(controls.norm()),
        "control_centered_norm": float(centered.norm()),
        "control_pairwise_rms": float(centered.pow(2).mean().sqrt()),
        "slow_lr": 0.0,
        "fast_lr": fast_scheduler.get_last_lr()[0],
    }


def phase_a_step(
    model,
    prompt_model,
    current_tensors,
    replay_tensors,
    preserve_targets,
    barycenter,
    slow_optimizer,
    fast_optimizer,
    slow_scheduler,
    fast_scheduler,
    normalization,
    old_face_mask,
    args,
):
    slow_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    fast_parameters = list(prompt_model.parameters())
    slow_optimizer.zero_grad(set_to_none=True)
    fast_optimizer.zero_grad(set_to_none=True)
    model.train()
    prompt_model.train()

    # One barycenter forward supplies exactly the same scalar objective to
    # slow weights and newly introduced controls.
    means, detached, _ = preference_forward(
        model, prompt_model, current_tensors, replay_tensors, preserve_targets,
        barycenter, len(barycenter),
        args.condition_chunk,
    )
    # Phase A has only one preference, so update the detached cumulative
    # statistics from this already-computed forward before scalarization. This
    # prevents an exact-zero initial FKL from making the next step divide by
    # only epsilon, and still requires no extra model evaluation.
    running_min_before = normalization["running_min"].clone()
    running_max_before = normalization["running_max"].clone()
    update_running_normalization(normalization, detached)
    scalar, normalized = stch_scalar(
        means, barycenter, normalization, args.stch_mu,
        normalize=len(barycenter) > 1,
    )
    scalar.backward()

    allreduce_grads(slow_parameters, average=False)
    allreduce_grads(fast_parameters, average=False)
    # ``old_face_mask`` is the caller-computed frozen-control mask.  In the
    # vertex-preserving mode it can contain only pure vertices even when old
    # face interior controls are trainable, so always apply it.
    if torch.any(old_face_mask):
        mask_old_face_gradient(prompt_model, old_face_mask)
    slow_grad_norm = torch.nn.utils.clip_grad_norm_(slow_parameters, args.max_grad_norm)
    fast_grad_norm = torch.nn.utils.clip_grad_norm_(fast_parameters, args.max_grad_norm)
    slow_optimizer.step()
    fast_optimizer.step()
    slow_scheduler.step()
    fast_scheduler.step()

    ideal_next = 0.9 * normalization["running_min"]
    scales_next = (
        normalization["running_max"] - ideal_next
    ).clamp_min(normalization["range_floor"])
    controls = prompt_model.controls.detach().float()
    centered = controls - controls.mean(0, keepdim=True)
    return {
        "phase": "A_joint_barycenter",
        "task_loss_by_preference": [detached.cpu().tolist()],
        "normalized_task_loss_by_preference": [normalized.detach().cpu().tolist()],
        "normalization_ideal": [0.0] * detached.shape[0],
        "normalization_running_min_before": running_min_before.tolist(),
        "normalization_running_max_before": running_max_before.tolist(),
        "normalization_running_min_used": normalization["running_min"].tolist(),
        "normalization_running_max_used": normalization["running_max"].tolist(),
        "normalization_ideal_next": ideal_next.tolist(),
        "normalization_scales_next": scales_next.tolist(),
        "normalization_initialized": normalization["initialized"],
        "slow_grad_norm": float(slow_grad_norm),
        "fast_grad_norm": float(fast_grad_norm),
        "reader_scale_mean": float(prompt_model.reader.layer_scales.detach().mean())
        if hasattr(prompt_model, "reader") else None,
        "reader_scale_min": float(prompt_model.reader.layer_scales.detach().min())
        if hasattr(prompt_model, "reader") else None,
        "reader_scale_max": float(prompt_model.reader.layer_scales.detach().max())
        if hasattr(prompt_model, "reader") else None,
        "reader_scale_lr": (
            fast_optimizer.param_groups[-1]["lr"]
            if hasattr(prompt_model, "reader") else None
        ),
        "control_norm": float(controls.norm()),
        "control_centered_norm": float(centered.norm()),
        "control_pairwise_rms": float(centered.pow(2).mean().sqrt()),
        "slow_lr": slow_scheduler.get_last_lr()[0],
        "fast_lr": fast_scheduler.get_last_lr()[0],
    }


def phase_b_step(
    model,
    prompt_model,
    current_tensors,
    replay_tensors,
    preserve_targets,
    preferences,
    fast_optimizer,
    fast_scheduler,
    normalization,
    old_face_mask,
    args,
    fixed_scales=None,
    taskwise_dual=None,
    taskwise_violation_ema=None,
    dual_probe_current_tensors=None,
    dual_probe_replay_tensors=None,
):
    """Freeze theta and fit only new/mixed simplex controls at random prefs."""
    model.requires_grad_(False)
    # Keep training mode so Transformers gradient checkpointing remains
    # active while gradients flow through the frozen backbone to soft inputs.
    # Qwen3-1.7B has attention_dropout=0, so this does not change the function.
    model.train()
    prompt_model.requires_grad_(True)
    prompt_model.train()
    fast_optimizer.zero_grad(set_to_none=True)
    detached_losses = []
    normalized_losses = []
    weighted_batch_regrets = []
    batch_regret_penalties = []
    batch_regret_gradient_weights = []
    taskwise_regrets = []
    taskwise_targets = []
    taskwise_violations = []
    taskwise_penalties = []
    taskwise_gradient_weights = []
    taskwise_constraint_masks = []
    taskwise_violation_sums = torch.zeros_like(taskwise_dual) if taskwise_dual is not None else None
    taskwise_violation_counts = torch.zeros_like(taskwise_dual) if taskwise_dual is not None else None
    apex_probe_delta = None
    apex_probe_violation = None
    if taskwise_dual is not None and dual_probe_current_tensors is not None:
        if not args.apex_specific_taskwise_dual:
            raise ValueError(
                "cross-fit dual probes require --apex-specific-taskwise-dual"
            )
        task_count = preferences.shape[-1]
        probe_vertices = torch.eye(
            task_count, device=preferences.device, dtype=preferences.dtype,
        )
        with torch.no_grad():
            _, _, _, probe_diagnostics = preference_forward_batch(
                model, prompt_model,
                dual_probe_current_tensors, dual_probe_replay_tensors,
                None, probe_vertices, task_count,
                args.condition_chunk, args.condition_token_budget,
                args.relative_softplus_tau,
            )
        if probe_diagnostics is None:
            raise RuntimeError(
                "cross-fit dual probe requires relative Slow diagnostics"
            )
        probe_delta_matrix = probe_diagnostics[
            "relative_delta_mean_by_preference"
        ]
        apex_probe_delta = probe_delta_matrix.diagonal()
        apex_probe_violation = (
            apex_probe_delta + args.taskwise_regret_margin
        )
        taskwise_violation_sums.copy_(apex_probe_violation)
        taskwise_violation_counts.fill_(1)
    for pref_start in range(0, len(preferences), args.preference_chunk):
        pref_end = min(pref_start + args.preference_chunk, len(preferences))
        pref_chunk = preferences[pref_start:pref_end]
        # First obtain the exact global task means without retaining model
        # activations.  The STCH derivative with respect to those means is
        # tiny (P x T); a second fused pass can then stream exact VJPs one row
        # block at a time.  This is numerically the same objective as one huge
        # backward but does not retain the whole Cartesian-product graph.
        with torch.no_grad():
            _, detached_batch, global_counts, relative_diagnostics = preference_forward_batch(
                model, prompt_model, current_tensors, replay_tensors,
                preserve_targets, pref_chunk, preferences.shape[-1],
                args.condition_chunk, args.condition_token_budget,
                args.relative_softplus_tau,
            )
        if args.scalarization == "weighted_sum" or args.disable_stch_normalization:
            scales = torch.ones_like(detached_batch[0])
            normalized_batch = detached_batch
        elif fixed_scales is not None and preferences.shape[-1] > 1:
            scales = fixed_scales.to(detached_batch.device)
            normalized_batch = detached_batch / scales
        else:
            normalized_batch = []
            for means in detached_batch:
                _, normalized = stch_scalar(
                    means, pref_chunk[len(normalized_batch)], normalization,
                    args.stch_mu, normalize=preferences.shape[-1] > 1,
                )
                normalized_batch.append(normalized)
            normalized_batch = torch.stack(normalized_batch)
            if preferences.shape[-1] > 1 and normalization["initialized"]:
                ideal = 0.9 * normalization["running_min"].to(
                    detached_batch.device,
                )
                scales = (
                    normalization["running_max"].to(detached_batch.device)
                    - ideal
                ).clamp_min(normalization["range_floor"])
            else:
                scales = torch.ones_like(detached_batch[0])
        if args.scalarization == "weighted_sum":
            task_coefficients = pref_chunk / len(preferences)
        else:
            stch_probabilities = torch.softmax(
                pref_chunk * normalized_batch / args.stch_mu, dim=-1,
            )
            task_coefficients = (
                stch_probabilities * pref_chunk / scales.unsqueeze(0)
                / len(preferences)
            )
        raw_task_coefficients = None
        if args.batch_regret_weight > 0:
            if relative_diagnostics is None:
                raise RuntimeError(
                    "batch regret margin requires relative Slow diagnostics"
                )
            raw_delta_means = relative_diagnostics[
                "relative_delta_mean_by_preference"
            ]
            weighted_regret = (pref_chunk * raw_delta_means).sum(dim=-1)
            margin_argument = (
                weighted_regret + args.batch_regret_margin
            ) / args.batch_regret_tau
            gradient_weight = torch.sigmoid(margin_argument)
            penalty = (
                args.batch_regret_weight * args.batch_regret_tau
                * torch.nn.functional.softplus(margin_argument)
            )
            raw_task_coefficients = (
                args.batch_regret_weight * gradient_weight.unsqueeze(-1)
                * pref_chunk / len(preferences)
            )
            weighted_batch_regrets.extend(weighted_regret.unbind(0))
            batch_regret_penalties.extend(penalty.unbind(0))
            batch_regret_gradient_weights.extend(gradient_weight.unbind(0))
        if taskwise_dual is not None:
            if relative_diagnostics is None:
                raise RuntimeError(
                    "taskwise constraint requires relative Slow diagnostics"
                )
            raw_delta_means = relative_diagnostics[
                "relative_delta_mean_by_preference"
            ]
            if args.apex_specific_taskwise_dual:
                # One diagonal constraint per available exact vertex:
                #     Delta_t(e_t) <= -m_t.
                # Interior preferences do not update either the apex dual or
                # the pure vertex control.  This prevents easy interior points
                # from averaging away a failed task apex.
                apex_pair_mask = pref_chunk.ge(1.0 - 1e-6)
                target = torch.full_like(
                    raw_delta_means, -args.taskwise_regret_margin,
                )
                violation = raw_delta_means - target
            else:
                # Every task must improve in proportion to its requested weight:
                #     Delta_{lambda,t} <= -m * lambda_t.
                apex_pair_mask = torch.ones_like(pref_chunk, dtype=torch.bool)
                target = -args.taskwise_regret_margin * pref_chunk
                violation = raw_delta_means - target
            gradient_weight = torch.sigmoid(
                violation / args.taskwise_regret_tau,
            ) * apex_pair_mask
            dual_weight = taskwise_dual.unsqueeze(0)
            penalty = (
                dual_weight * args.taskwise_regret_tau
                * torch.nn.functional.softplus(
                    violation / args.taskwise_regret_tau,
                )
            ) * apex_pair_mask
            constraint_counts = apex_pair_mask.sum(0).clamp_min(1)
            taskwise_coefficients = (
                dual_weight * gradient_weight
                / constraint_counts.unsqueeze(0)
            )
            if not args.apex_specific_taskwise_dual:
                taskwise_coefficients = taskwise_coefficients / len(preferences)
            raw_task_coefficients = (
                taskwise_coefficients if raw_task_coefficients is None
                else raw_task_coefficients + taskwise_coefficients
            )
            if dual_probe_current_tensors is None:
                taskwise_violation_sums.add_(
                    (violation * apex_pair_mask).sum(dim=0)
                )
                taskwise_violation_counts.add_(apex_pair_mask.sum(dim=0))
            taskwise_regrets.extend(raw_delta_means.unbind(0))
            taskwise_targets.extend(target.unbind(0))
            taskwise_violations.extend(violation.unbind(0))
            taskwise_penalties.extend(penalty.unbind(0))
            taskwise_gradient_weights.extend(gradient_weight.unbind(0))
            taskwise_constraint_masks.extend(apex_pair_mask.unbind(0))
        preference_backward_streaming(
            model, prompt_model, current_tensors, replay_tensors,
            preserve_targets, pref_chunk, task_coefficients, global_counts,
            args.condition_chunk, args.condition_token_budget,
            args.relative_softplus_tau,
            raw_task_coefficients,
            isolate_vertex_gradients=args.apex_specific_taskwise_dual,
        )
        detached_losses.extend(detached_batch.unbind(0))
        normalized_losses.extend(normalized_batch.detach().unbind(0))
    stacked_detached = torch.stack(detached_losses)
    stacked_normalized = torch.stack(normalized_losses)
    if args.scalarization == "weighted_sum":
        stch_losses = (preferences * stacked_detached).sum(dim=-1)
    else:
        stch_losses = args.stch_mu * torch.logsumexp(
            preferences * stacked_normalized / args.stch_mu, dim=-1,
        )
    running_min_used = normalization["running_min"].clone()
    running_max_used = normalization["running_max"].clone()
    if (
        fixed_scales is None and args.scalarization != "weighted_sum"
        and not args.disable_stch_normalization
    ):
        update_running_normalization(normalization, stacked_detached)
    ideal_next = 0.9 * normalization["running_min"]
    scales_next = (
        normalization["running_max"] - ideal_next
    ).clamp_min(normalization["range_floor"])
    fast_parameters = list(prompt_model.parameters())
    allreduce_grads(fast_parameters, average=False)
    if torch.any(old_face_mask):
        mask_old_face_gradient(prompt_model, old_face_mask)
    fast_grad_norm = torch.nn.utils.clip_grad_norm_(
        fast_parameters, args.max_grad_norm,
    )
    fast_optimizer.step()
    fast_scheduler.step()
    fast_optimizer.zero_grad(set_to_none=True)
    taskwise_dual_before = taskwise_dual.detach().clone() if taskwise_dual is not None else None
    taskwise_violation_ema_before = (
        taskwise_violation_ema.detach().clone()
        if taskwise_violation_ema is not None else None
    )
    if taskwise_dual is not None:
        with torch.no_grad():
            active = taskwise_violation_counts > 0
            average_violation = taskwise_violation_sums / taskwise_violation_counts.clamp_min(1)
            if taskwise_violation_ema is not None:
                uninitialized = torch.isnan(taskwise_violation_ema) & active
                taskwise_violation_ema[uninitialized] = average_violation[uninitialized]
                initialized = active & ~uninitialized
                taskwise_violation_ema[initialized] = (
                    args.taskwise_dual_ema_decay
                    * taskwise_violation_ema[initialized]
                    + (1.0 - args.taskwise_dual_ema_decay)
                    * average_violation[initialized]
                )
                dual_violation = taskwise_violation_ema
            else:
                dual_violation = average_violation
            taskwise_dual[active] = (
                taskwise_dual[active]
                + args.taskwise_dual_lr * dual_violation[active]
            )
            taskwise_dual.clamp_(0.0, args.taskwise_dual_max)
    model.requires_grad_(True)
    controls = prompt_model.controls.detach().float()
    centered = controls - controls.mean(0, keepdim=True)
    return {
        "phase": "B_fast_random_simplex",
        "task_loss_by_preference": torch.stack(detached_losses).cpu().tolist(),
        "normalized_task_loss_by_preference": torch.stack(
            normalized_losses,
        ).cpu().tolist(),
        "stch_loss_by_preference": stch_losses.detach().cpu().tolist(),
        "stch_loss_mean": float(stch_losses.detach().mean()),
        "weighted_batch_regret_by_preference": (
            torch.stack(weighted_batch_regrets).detach().cpu().tolist()
            if weighted_batch_regrets else None
        ),
        "batch_regret_penalty_by_preference": (
            torch.stack(batch_regret_penalties).detach().cpu().tolist()
            if batch_regret_penalties else None
        ),
        "batch_regret_gradient_weight_by_preference": (
            torch.stack(batch_regret_gradient_weights).detach().cpu().tolist()
            if batch_regret_gradient_weights else None
        ),
        "batch_regret_negative_fraction": (
            float((torch.stack(weighted_batch_regrets) < 0).float().mean())
            if weighted_batch_regrets else None
        ),
        "taskwise_regret_by_preference": (
            torch.stack(taskwise_regrets).detach().cpu().tolist()
            if taskwise_regrets else None
        ),
        "taskwise_target_by_preference": (
            torch.stack(taskwise_targets).detach().cpu().tolist()
            if taskwise_targets else None
        ),
        "taskwise_violation_by_preference": (
            torch.stack(taskwise_violations).detach().cpu().tolist()
            if taskwise_violations else None
        ),
        "taskwise_constraint_mode": (
            "apex_specific_diagonal"
            if args.apex_specific_taskwise_dual else
            "all_preference_task_pairs"
        ),
        "apex_specific_taskwise_dual": bool(
            args.apex_specific_taskwise_dual
        ),
        "apex_dual_update_source": (
            "held_out_probe"
            if dual_probe_current_tensors is not None else
            "primal_training_batch"
        ),
        "apex_probe_delta": (
            apex_probe_delta.detach().cpu().tolist()
            if apex_probe_delta is not None else None
        ),
        "apex_probe_violation": (
            apex_probe_violation.detach().cpu().tolist()
            if apex_probe_violation is not None else None
        ),
        "taskwise_all_feasible_fraction": (
            float((
                ((torch.stack(taskwise_violations) <= 0)
                 | ~torch.stack(taskwise_constraint_masks)).all(dim=-1)
            )[torch.stack(taskwise_constraint_masks).any(dim=-1)].float().mean())
            if taskwise_violations else None
        ),
        "taskwise_constraint_feasible_fraction": (
            float((
                (torch.stack(taskwise_violations) <= 0)
                & torch.stack(taskwise_constraint_masks)
            ).float().sum() / torch.stack(taskwise_constraint_masks).float().sum().clamp_min(1))
            if taskwise_violations else None
        ),
        "taskwise_constraint_mask_by_preference": (
            torch.stack(taskwise_constraint_masks).detach().cpu().tolist()
            if taskwise_constraint_masks else None
        ),
        "taskwise_penalty_by_preference": (
            torch.stack(taskwise_penalties).detach().cpu().tolist()
            if taskwise_penalties else None
        ),
        "taskwise_gradient_weight_by_preference": (
            torch.stack(taskwise_gradient_weights).detach().cpu().tolist()
            if taskwise_gradient_weights else None
        ),
        "taskwise_dual_before": (
            taskwise_dual_before.cpu().tolist()
            if taskwise_dual_before is not None else None
        ),
        "taskwise_dual_after": (
            taskwise_dual.detach().cpu().tolist()
            if taskwise_dual is not None else None
        ),
        "taskwise_violation_ema_before": (
            taskwise_violation_ema_before.cpu().tolist()
            if taskwise_violation_ema_before is not None else None
        ),
        "taskwise_violation_ema_after": (
            taskwise_violation_ema.detach().cpu().tolist()
            if taskwise_violation_ema is not None else None
        ),
        "scalarization": args.scalarization,
        "normalization_ideal": [0.0] * len(preferences[0]),
        "normalization_running_min_used": running_min_used.tolist(),
        "normalization_running_max_used": running_max_used.tolist(),
        "normalization_ideal_next": ideal_next.tolist(),
        "normalization_scales_next": scales_next.tolist(),
        "normalization_initialized": normalization["initialized"],
        "fast_fixed_scales": (
            fixed_scales.detach().cpu().tolist()
            if fixed_scales is not None else None
        ),
        "slow_grad_norm": 0.0,
        "fast_grad_norm": float(fast_grad_norm),
        "reader_scale_mean": float(prompt_model.reader.layer_scales.detach().mean())
        if hasattr(prompt_model, "reader") else None,
        "reader_scale_min": float(prompt_model.reader.layer_scales.detach().min())
        if hasattr(prompt_model, "reader") else None,
        "reader_scale_max": float(prompt_model.reader.layer_scales.detach().max())
        if hasattr(prompt_model, "reader") else None,
        "reader_scale_lr": (
            fast_optimizer.param_groups[-1]["lr"]
            if hasattr(prompt_model, "reader") else None
        ),
        "control_norm": float(controls.norm()),
        "control_centered_norm": float(centered.norm()),
        "control_pairwise_rms": float(centered.pow(2).mean().sqrt()),
        "slow_lr": 0.0,
        "fast_lr": fast_scheduler.get_last_lr()[0],
        **({
            key: value.cpu().tolist()
            for key, value in relative_diagnostics.items()
        } if relative_diagnostics is not None else {}),
    }


@torch.no_grad()
def fast_only_calibration_step(
    model, prompt_model, current_tensors, replay_tensors, preserve_targets,
    preferences, args,
):
    """Measure objective scales without changing Slow or Fast parameters."""
    model.requires_grad_(False)
    model.eval()
    prompt_model.requires_grad_(False)
    prompt_model.eval()
    detached_losses = []
    for pref_start in range(0, len(preferences), args.preference_chunk):
        pref_end = min(pref_start + args.preference_chunk, len(preferences))
        pref_chunk = preferences[pref_start:pref_end]
        _, detached, _, relative_diagnostics = preference_forward_batch(
            model, prompt_model, current_tensors, replay_tensors,
            preserve_targets, pref_chunk, preferences.shape[-1],
            args.condition_chunk, args.condition_token_budget,
            args.relative_softplus_tau,
        )
        detached_losses.extend(detached.unbind(0))
    stacked = torch.stack(detached_losses)
    return {
        "phase": "fast_only_scale_calibration",
        "task_loss_by_preference": stacked.cpu().tolist(),
        "normalized_task_loss_by_preference": None,
        "calibration_task_means": stacked.mean(0).cpu().tolist(),
        "fast_fixed_scales": None,
        "slow_grad_norm": 0.0,
        "fast_grad_norm": 0.0,
        "reader_scale_mean": float(prompt_model.reader.layer_scales.detach().mean())
        if hasattr(prompt_model, "reader") else None,
        "reader_scale_min": float(prompt_model.reader.layer_scales.detach().min())
        if hasattr(prompt_model, "reader") else None,
        "reader_scale_max": float(prompt_model.reader.layer_scales.detach().max())
        if hasattr(prompt_model, "reader") else None,
        "reader_scale_lr": 0.0,
        "control_norm": float(prompt_model.controls.detach().float().norm()),
        "control_centered_norm": float((
            prompt_model.controls.detach().float()
            - prompt_model.controls.detach().float().mean(0, keepdim=True)
        ).norm()),
        "control_pairwise_rms": float((
            prompt_model.controls.detach().float()
            - prompt_model.controls.detach().float().mean(0, keepdim=True)
        ).pow(2).mean().sqrt()),
        "slow_lr": 0.0,
        "fast_lr": 0.0,
        **({
            key: value.cpu().tolist()
            for key, value in relative_diagnostics.items()
        } if relative_diagnostics is not None else {}),
    }


def joint_projection_step(
    model,
    prompt_model,
    current_tensors,
    replay_tensors,
    preserve_targets,
    preferences,
    slow_optimizer,
    fast_optimizer,
    slow_scheduler,
    fast_scheduler,
    fixed_fast_scale,
    update_fast,
    diagnose_fast,
    args,
):
    """Simultaneously update slow and fast weights from the same forwards.

    Slow uses the current-task gradient projected onto the replay-safe
    half-space. Fast uses first-batch fixed-scale STCH over all task losses.
    Gradients are accumulated over four preferences before a single update.
    """
    model.requires_grad_(True)
    model.train()
    prompt_model.requires_grad_(update_fast)
    prompt_model.train()
    slow_optimizer.zero_grad(set_to_none=True)
    fast_optimizer.zero_grad(set_to_none=True)
    slow_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    fast_parameters = [
        parameter for parameter in prompt_model.parameters()
        if parameter.requires_grad
    ]
    acquire_objectives = []
    preserve_objectives = []
    preference_means = []
    detached_losses = []

    # Retain the four preference graphs and differentiate their exact means
    # only once per objective.  This is mathematically identical to summing
    # four per-preference VJPs, while reducing 12 graph traversals to three.
    for preference in preferences:
        means, detached, _ = preference_forward(
            model, prompt_model, current_tensors, replay_tensors,
            preserve_targets, preference, len(preference),
            args.condition_chunk,
        )
        acquire = means[-1]
        preserve = means[:-1].mean()
        acquire_objectives.append(acquire)
        preserve_objectives.append(preserve)
        preference_means.append(means)
        detached_losses.append(detached)

    stacked_detached = torch.stack(detached_losses)
    calibration_s_p = stacked_detached[:, :-1].mean().clamp_min(1e-12)
    calibration_s_a = stacked_detached[:, -1].mean().clamp_min(1e-12)
    fast_objectives = []
    normalized_losses = []
    scales = None
    if update_fast:
        if fixed_fast_scale["values"] is None:
            raise RuntimeError("Fast update started before fixed-scale calibration")
        scales = fixed_fast_scale["values"].to(preferences.device)
        for means, preference in zip(preference_means, preferences):
            normalized = means / scales
            weighted = preference * normalized
            fast_objectives.append(
                args.stch_mu * torch.logsumexp(
                    weighted / args.stch_mu, dim=0,
                )
            )
            normalized_losses.append(normalized.detach())

    acquire_mean = torch.stack(acquire_objectives).mean()
    preserve_mean = torch.stack(preserve_objectives).mean()
    ga = torch.autograd.grad(
        acquire_mean, slow_parameters, retain_graph=True, allow_unused=True,
    )
    gp = torch.autograd.grad(
        preserve_mean, slow_parameters, retain_graph=update_fast,
        allow_unused=True,
    )
    acquire_grads = accumulate_grad_list(None, ga, slow_parameters)
    preserve_grads = accumulate_grad_list(None, gp, slow_parameters)
    fast_grads = []
    fast_acquire_grads = []
    fast_preserve_grads = []
    if update_fast:
        fast_mean = torch.stack(fast_objectives).mean()
        if diagnose_fast:
            gfa = torch.autograd.grad(
                acquire_mean, fast_parameters, retain_graph=True,
                allow_unused=True,
            )
            gfp = torch.autograd.grad(
                preserve_mean, fast_parameters, retain_graph=True,
                allow_unused=True,
            )
            fast_acquire_grads = accumulate_grad_list(
                None, gfa, fast_parameters,
            )
            fast_preserve_grads = accumulate_grad_list(
                None, gfp, fast_parameters,
            )
        gf = torch.autograd.grad(
            fast_mean, fast_parameters, retain_graph=False, allow_unused=True,
        )
        fast_grads = accumulate_grad_list(None, gf, fast_parameters)

    # distributed_task_means makes each local gradient a contribution to the
    # exact global mean, hence use SUM rather than world-size averaging.
    allreduce_grad_list(acquire_grads)
    allreduce_grad_list(preserve_grads)
    if update_fast:
        allreduce_grad_list(fast_grads)
        if diagnose_fast:
            allreduce_grad_list(fast_acquire_grads)
            allreduce_grad_list(fast_preserve_grads)

    dot = torch.zeros((), device=preferences.device, dtype=torch.float32)
    acquire_sq = torch.zeros_like(dot)
    preserve_sq = torch.zeros_like(dot)
    for ga, gp in zip(acquire_grads, preserve_grads):
        dot.add_((ga.float() * gp.float()).sum())
        acquire_sq.add_(ga.float().square().sum())
        preserve_sq.add_(gp.float().square().sum())
    conflicting = bool(dot.item() < 0.0)
    coefficient = dot / (preserve_sq + 1e-12) if conflicting else dot.new_zeros(())
    projected_sq = torch.zeros_like(dot)
    for parameter, ga, gp in zip(
        slow_parameters, acquire_grads, preserve_grads,
    ):
        direction = ga - coefficient.to(dtype=ga.dtype) * gp if conflicting else ga
        parameter.grad = direction
        projected_sq.add_(direction.float().square().sum())
    if update_fast:
        for parameter, grad in zip(fast_parameters, fast_grads):
            parameter.grad = grad

    cosine = dot / (acquire_sq.sqrt() * preserve_sq.sqrt() + 1e-12)
    fast_objective_cosine = None
    fast_acquire_norm = None
    fast_preserve_norm = None
    if diagnose_fast:
        fast_dot = torch.zeros_like(dot)
        fast_acquire_sq = torch.zeros_like(dot)
        fast_preserve_sq = torch.zeros_like(dot)
        for gfa, gfp in zip(fast_acquire_grads, fast_preserve_grads):
            fast_dot.add_((gfa.float() * gfp.float()).sum())
            fast_acquire_sq.add_(gfa.float().square().sum())
            fast_preserve_sq.add_(gfp.float().square().sum())
        fast_objective_cosine = float(
            fast_dot / (
                fast_acquire_sq.sqrt() * fast_preserve_sq.sqrt() + 1e-12
            )
        )
        fast_acquire_norm = float(fast_acquire_sq.sqrt())
        fast_preserve_norm = float(fast_preserve_sq.sqrt())
    slow_grad_norm_clipped = torch.nn.utils.clip_grad_norm_(
        slow_parameters, args.max_grad_norm,
    )
    fast_grad_norm_clipped = (
        torch.nn.utils.clip_grad_norm_(fast_parameters, args.max_grad_norm)
        if update_fast else torch.zeros((), device=preferences.device)
    )
    slow_optimizer.step()
    if update_fast:
        fast_optimizer.step()
    slow_scheduler.step()
    if update_fast:
        fast_scheduler.step()

    controls = prompt_model.controls.detach().float()
    centered = controls - controls.mean(0, keepdim=True)
    with torch.no_grad():
        vertices = torch.eye(
            prompt_model.num_tasks, device=preferences.device,
        )
        vertex_prompts = prompt_model(vertices)
        endpoint_delta = vertex_prompts[0] - vertex_prompts[-1]
        endpoint_bf16_diff = (
            vertex_prompts[0].to(torch.bfloat16)
            != vertex_prompts[-1].to(torch.bfloat16)
        ).float().mean()
    stacked = stacked_detached
    return {
        "phase": (
            "joint_projection_fixedscale_stch"
            if update_fast else "slow_only_scale_calibration"
        ),
        "task_loss_by_preference": stacked.cpu().tolist(),
        "normalized_task_loss_by_preference": (
            torch.stack(normalized_losses).cpu().tolist()
            if normalized_losses else None
        ),
        "stch_loss_by_preference": (
            torch.stack([value.detach() for value in fast_objectives]).cpu().tolist()
            if fast_objectives else None
        ),
        "fast_fixed_scales": (
            scales.detach().cpu().tolist() if scales is not None else None
        ),
        "fast_scale_preserve": (
            float(scales[:-1].mean()) if scales is not None else None
        ),
        "fast_scale_acquire": (
            float(scales[-1]) if scales is not None else None
        ),
        "calibration_batch_preserve": float(calibration_s_p),
        "calibration_batch_acquire": float(calibration_s_a),
        "fast_updated": bool(update_fast),
        "fast_objective_gradient_cosine": fast_objective_cosine,
        "fast_acquire_grad_norm_diagnostic": fast_acquire_norm,
        "fast_preserve_grad_norm_diagnostic": fast_preserve_norm,
        "endpoint_prompt_delta_rms_fp32": float(
            endpoint_delta.float().square().mean().sqrt()
        ),
        "endpoint_prompt_bf16_distinct_fraction": float(endpoint_bf16_diff),
        "acquire_loss_mean": float(stacked[:, -1].mean()),
        "preserve_loss_mean": float(stacked[:, :-1].mean()),
        "slow_acquire_grad_norm": float(acquire_sq.sqrt()),
        "slow_preserve_grad_norm": float(preserve_sq.sqrt()),
        "slow_grad_dot": float(dot),
        "slow_grad_cosine": float(cosine),
        "slow_projection_triggered": conflicting,
        "slow_projection_coefficient": float(coefficient),
        "slow_projected_grad_norm": float(projected_sq.sqrt()),
        "slow_grad_norm_before_clip": float(slow_grad_norm_clipped),
        "fast_grad_norm_before_clip": float(fast_grad_norm_clipped),
        "control_norm": float(controls.norm()),
        "control_centered_norm": float(centered.norm()),
        "control_pairwise_rms": float(centered.pow(2).mean().sqrt()),
        "slow_lr": slow_scheduler.get_last_lr()[0],
        "fast_lr": fast_scheduler.get_last_lr()[0],
    }


def initialize_prompt_model(args, model, tokenizer, device):
    if args.hidden_residual:
        prompt = SimplexHiddenResidual(
            model, args.stage + 1, args.bezier_degree, args.prompt_length,
            args.reader_scale_init, args.reader_o_lora_rank,
            layer_specific_controls=args.layer_specific_controls,
            trainable_layer_alpha=args.trainable_layer_alpha,
            current_slow_anchor=args.current_slow_anchor,
        ).to(device)
        if args.stage == 0 or args.fresh_fast_controls:
            return prompt
        if args.previous_prompt is None:
            raise ValueError("hidden residual stage > 0 requires --previous-prompt")
        source = args.previous_prompt
        source_dir = source if source.is_dir() else source.parent
        generator_path = (
            source / "layer_simplex_residual_memory.pt"
            if source.is_dir() and args.layer_specific_controls else
            source / "simplex_residual_memory.pt"
            if source.is_dir() else source
        )
        reader_path = source_dir / "memory_reader.pt"
        if not generator_path.exists() or not reader_path.exists():
            raise ValueError(
                f"missing hidden-memory state: {generator_path} / {reader_path}"
            )
        state = torch.load(generator_path, map_location="cpu", weights_only=True)
        old_tasks = int(state["multi_indices"].shape[1])
        expected = args.stage + 1 if args.load_same_stage_prompt else args.stage
        if old_tasks != expected:
            raise ValueError(
                f"hidden-memory state has {old_tasks} task dimensions, expected {expected}"
            )
        if args.load_same_stage_prompt:
            prompt.generator.load_state_dict(state)
        else:
            # Embed the old simplex exactly as the new face lambda_t=0. New
            # controls start at the old barycentric memory; the exact current
            # vertex is subsequently zeroed by --current-slow-anchor.
            old_indices = state["multi_indices"].long()
            old_controls = state["controls"].float()
            old_map = {tuple(v.tolist()): i for i, v in enumerate(old_indices)}
            old_center = old_controls.mean(0)
            with torch.no_grad():
                prompt.controls.copy_(
                    old_center.unsqueeze(0).expand_as(prompt.controls)
                )
                for new_index, alpha in enumerate(prompt.multi_indices.cpu()):
                    if int(alpha[-1]) == 0:
                        key = tuple(alpha[:-1].tolist())
                        prompt.controls[new_index].copy_(old_controls[old_map[key]])
        prompt.reader.load_state_dict(torch.load(
            reader_path, map_location="cpu", weights_only=True,
        ))
        prompt.recursive_verification = {
            "hidden_memory_loaded": True,
            "same_stage": bool(args.load_same_stage_prompt),
            "source_task_count": old_tasks,
            "target_task_count": args.stage + 1,
        }
        return prompt
    if args.stage == 0 or args.fresh_fast_controls:
        return SimplexBezierPrompt(
            args.stage + 1, args.bezier_degree, args.prompt_length,
            int(model.config.hidden_size), random_init=True,
        ).to(device)
    if args.previous_prompt is None or not args.previous_prompt.exists():
        raise ValueError("stage > 0 requires --previous-prompt")
    state = torch.load(args.previous_prompt, map_location="cpu", weights_only=True)
    state_task_count = int(state["multi_indices"].shape[1])
    if args.load_same_stage_prompt:
        if state_task_count != args.stage + 1:
            raise ValueError(
                "--load-same-stage-prompt expected "
                f"{args.stage + 1} task coordinates, found {state_task_count}"
            )
        prompt = SimplexBezierPrompt(
            state_task_count, args.bezier_degree, args.prompt_length,
            int(model.config.hidden_size),
            residual_scale=float(args.fast_residual_scale),
        )
        prompt.load_state_dict(state)
        prompt.recursive_verification = {
            "same_stage_prompt_loaded": True,
            "task_count": state_task_count,
        }
        return prompt.to(device)
    # Preserve the old face exactly and initialize each newly introduced
    # control independently like torch.nn.Embedding.
    old_task_count = state_task_count
    old_prompt = SimplexBezierPrompt(
        old_task_count, args.bezier_degree, args.prompt_length,
        int(model.config.hidden_size),
        residual_scale=float(args.fast_residual_scale),
    )
    old_prompt.load_state_dict(state)
    expanded = SimplexBezierPrompt.expand_from_state(
        state, args.stage + 1, random_init=True,
    )
    expanded.recursive_verification = verify_recursive_expansion(
        old_prompt, expanded,
    )
    return expanded.to(device)


def load_frozen_previous_prompt(args, model, device):
    """Load the exact old-stage fast-memory chart used by the FKL teacher."""
    if args.stage == 0:
        return None
    state = torch.load(args.previous_prompt, map_location="cpu", weights_only=True)
    old_task_count = int(state["multi_indices"].shape[1])
    old_prompt = SimplexBezierPrompt(
        old_task_count, args.bezier_degree, args.prompt_length,
        int(model.config.hidden_size),
    ).to(device)
    old_prompt.load_state_dict(state)
    old_prompt.requires_grad_(False)
    old_prompt.eval()
    return old_prompt


def main():
    global _ACTIVE_HIDDEN_RESIDUAL
    args = parse_args()
    if args.preference_count < 1 or args.preference_chunk < 1:
        raise ValueError("preference-count and preference-chunk must be positive")
    if args.acquire_only_current_apex and (args.fast_only or args.joint_projection):
        raise ValueError(
            "--acquire-only-current-apex is incompatible with --fast-only "
            "and --joint-projection"
        )
    if args.current_apex_warmup_steps < 0:
        raise ValueError("--current-apex-warmup-steps must be non-negative")
    if args.current_apex_warmup_steps and not args.fast_only:
        raise ValueError("--current-apex-warmup-steps requires --fast-only")
    if args.batch_regret_weight < 0 or args.batch_regret_tau <= 0:
        raise ValueError("batch regret weight must be non-negative and tau positive")
    if args.batch_regret_weight > 0 and args.relative_softplus_tau <= 0:
        raise ValueError("batch regret margin requires --relative-softplus-tau > 0")
    if args.taskwise_dual_lr < 0 or args.taskwise_regret_tau <= 0:
        raise ValueError("taskwise dual LR must be non-negative and tau positive")
    if not 0 <= args.taskwise_dual_ema_decay < 1:
        raise ValueError("taskwise dual EMA decay must be in [0, 1)")
    if args.taskwise_dual_lr > 0 and args.relative_softplus_tau <= 0:
        raise ValueError("taskwise constraint requires --relative-softplus-tau > 0")
    if args.stage != TASKS.index(args.task):
        raise ValueError("stage must match the canonical TRACE task order")
    # Compatibility fields consumed by shared preprocessing helpers.
    args.acquire_objective = "sft"
    args.max_completion_length = args.max_completion_length or MAX_COMPLETION[args.task]
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    rank = accelerator.process_index
    world = accelerator.num_processes
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    current_rows, rejected, replay_rows = load_cached_prepared_rows(
        args, tokenizer, accelerator,
    )
    if args.stage > 0 and not replay_rows:
        raise RuntimeError("continual stages require a non-empty replay buffer")
    replay_pools: dict[int, list[dict]] = {}
    for row in replay_rows:
        replay_pools.setdefault(int(row["task_id"]), []).append(row)
    expected_old_tasks = set(range(args.stage))
    if set(replay_pools) != expected_old_tasks:
        raise RuntimeError(
            f"replay tasks {sorted(replay_pools)} != expected {sorted(expected_old_tasks)}"
        )
    fixed_heldout_dual = bool(
        args.heldout_current_dual_probe_size
        or args.heldout_replay_dual_probe_size
    )
    if args.crossfit_dual_probe_size and fixed_heldout_dual:
        raise ValueError(
            "rotating cross-fit and fixed held-out dual probes are mutually exclusive"
        )
    if args.crossfit_dual_probe_size or fixed_heldout_dual:
        if not args.fast_only or not args.apex_specific_taskwise_dual:
            raise ValueError(
                "held-out dual probes require --fast-only and "
                "--apex-specific-taskwise-dual"
            )
        if args.preserve_objective != "sft":
            raise ValueError(
                "held-out dual probes currently require --preserve-objective sft"
            )
    if args.crossfit_dual_probe_size:
        if not 0 < args.crossfit_dual_probe_size < len(current_rows):
            raise ValueError("invalid current cross-fit probe size")
        if not 0 < args.crossfit_dual_probe_size < len(replay_rows):
            raise ValueError("invalid replay cross-fit probe size")
    if fixed_heldout_dual:
        if not 0 < args.heldout_current_dual_probe_size < len(current_rows):
            raise ValueError("invalid fixed current held-out probe size")
        if not 0 < args.heldout_replay_dual_probe_size < len(replay_rows):
            raise ValueError("invalid fixed replay held-out probe size")
    current_global_batch = (
        len(replay_rows) if args.current_batch_match_replay else args.global_batch
    )
    if current_global_batch < world:
        raise ValueError(
            f"global current batch {current_global_batch} must be >= world size {world}"
        )
    uneven_current_sharding = bool(current_global_batch % world)
    # Uneven logical global batches (for example 50 rows over four ranks) are
    # valid: current_indices_for_step() shards them round-robin and the loss
    # paths reduce sums together with their true row counts.  Requiring exact
    # divisibility here would silently change the requested experimental batch.
    local_batch = current_global_batch // world
    local_batch_min = current_global_batch // world
    local_batch_max = math.ceil(current_global_batch / world)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(device)
    preserve_teacher = None
    old_prompt_model = None
    if args.stage > 0 and args.preserve_objective == "topk_fkl":
        teacher_model = args.preserve_teacher_model or args.model
        preserve_teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to(device)
        preserve_teacher.requires_grad_(False)
        preserve_teacher.eval()
        preserve_teacher.config.use_cache = False
        old_prompt_model = (
            None if args.fast_only else
            load_frozen_previous_prompt(args, model, device)
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    model.config.use_cache = False
    if args.current_slow_anchor and not args.hidden_residual:
        raise ValueError(
            "--current-slow-anchor requires --hidden-residual; input-level "
            "soft tokens cannot provide an exact Slow-only endpoint"
        )
    if args.current_slow_anchor and args.current_apex_warmup_steps:
        raise ValueError(
            "the exact current Slow anchor is frozen and cannot use apex warmup"
        )
    prompt_model = initialize_prompt_model(args, model, tokenizer, device)
    if args.hidden_residual:
        if args.gradient_checkpointing:
            raise ValueError(
                "--hidden-residual currently requires --no-gradient-checkpointing "
                "so hook state is exact during backward recomputation"
            )
        _ACTIVE_HIDDEN_RESIDUAL = prompt_model
        if args.freeze_memory_reader:
            prompt_model.reader.requires_grad_(False)
            if prompt_model.layer_alpha_logits is not None:
                prompt_model.layer_alpha_logits.requires_grad_(False)
    prompt_model.residual_scale = float(args.fast_residual_scale)
    current_anchor_mask = prompt_model.multi_indices[:, -1].eq(
        args.bezier_degree
    )
    if args.current_slow_anchor:
        with torch.no_grad():
            prompt_model.controls[current_anchor_mask].zero_()

        def freeze_current_anchor_gradient(gradient):
            gradient = gradient.clone()
            gradient[current_anchor_mask] = 0
            return gradient

        prompt_model.controls.register_hook(freeze_current_anchor_gradient)
    initial_controls = prompt_model.controls.detach().float().clone()
    old_face_mask = (
        prompt_model.multi_indices[:, -1].eq(0)
        if args.stage > 0 else torch.zeros(
            len(prompt_model.controls), dtype=torch.bool, device=device,
        )
    )
    vertex_mask = prompt_model.multi_indices.eq(args.bezier_degree).any(dim=1)
    fast_frozen_mask = (
        vertex_mask if args.train_all_fast_controls and args.freeze_all_vertices else
        old_face_mask | vertex_mask if args.freeze_all_vertices else
        torch.zeros_like(old_face_mask) if args.train_all_fast_controls else
        old_face_mask
    )

    epochs = args.epochs or EPOCHS[args.task]
    steps = math.ceil(len(current_rows) / current_global_batch) * epochs
    if args.max_steps > 0:
        steps = min(steps, args.max_steps)
    slow_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.slow_lr, weight_decay=0.0, fused=True,
    )
    reader_scale_parameters = (
        [prompt_model.reader.layer_scales]
        if args.hidden_residual else []
    )
    reader_scale_ids = {id(p) for p in reader_scale_parameters}
    other_fast_parameters = [
        p for p in prompt_model.parameters()
        if p.requires_grad and id(p) not in reader_scale_ids
    ]
    def make_fast_optimizer():
        # Use fresh parameter-group dictionaries so the endpoint warmup and
        # Pareto phase can have independent Adam state.
        groups = [{"params": other_fast_parameters, "lr": args.fast_lr}]
        if reader_scale_parameters:
            groups.append({
                "params": reader_scale_parameters,
                "lr": (
                    args.reader_scale_lr
                    if args.reader_scale_lr is not None else args.fast_lr
                ),
            })
        return torch.optim.AdamW(groups, weight_decay=0.0, fused=True)

    fast_optimizer = make_fast_optimizer()
    slow_scheduler = make_scheduler(
        slow_optimizer, steps, min(args.warmup_steps, steps),
        args.lr_scheduler_type,
    )
    # Joint mode performs one simultaneous update per data step; the legacy
    # two-phase mode keeps one fast schedule across both passes.
    fast_only_calibration_steps = (
        min(args.fast_calibration_steps, max(0, steps - 1))
        if args.fast_only and args.stage > 0 else 0
    )
    fast_total_steps = (
        steps if args.acquire_only_current_apex else
        max(1, steps - fast_only_calibration_steps + args.current_apex_warmup_steps)
        if args.fast_only else
        max(1, steps - args.fast_calibration_steps)
        if args.joint_projection else 2 * steps
    )
    initial_fast_schedule_steps = (
        args.current_apex_warmup_steps
        if args.fast_only and args.current_apex_warmup_steps > 0
        else fast_total_steps
    )
    fast_scheduler = make_scheduler(
        fast_optimizer, initial_fast_schedule_steps,
        0 if args.joint_projection else min(
            args.warmup_steps, initial_fast_schedule_steps,
        ),
        args.lr_scheduler_type,
    )
    normalization = {
        "running_min": torch.zeros(args.stage + 1),
        "running_max": torch.zeros(args.stage + 1),
        "range_floor": float(args.range_floor),
        "initialized": False,
    }
    taskwise_dual = (
        torch.full(
            (args.stage + 1,), args.taskwise_dual_init, device=device,
        ) if args.taskwise_dual_lr > 0 else None
    )
    taskwise_violation_ema = (
        torch.full((args.stage + 1,), float("nan"), device=device)
        if taskwise_dual is not None else None
    )

    required = max(steps, args.current_apex_warmup_steps) * current_global_batch
    order: list[int] = []
    if not args.current_with_replacement:
        epoch = 0
        while len(order) < required:
            order.extend(bucketed_epoch_order(
                current_rows, args.seed, epoch, args.length_bucket_size,
            ))
            epoch += 1

    def current_indices_for_step(step_number: int) -> list[int]:
        if args.current_with_replacement:
            rng = random.Random(args.seed + 15485863 * step_number)
            global_indices = [
                rng.randrange(len(current_rows))
                for _ in range(current_global_batch)
            ]
        else:
            begin = (step_number - 1) * current_global_batch
            global_indices = order[begin:begin + current_global_batch]
        if uneven_current_sharding:
            return global_indices[rank::world]
        return global_indices[rank * local_batch:(rank + 1) * local_batch]

    def crossfit_rows_for_step(rows, step_number, salt):
        """Return disjoint primal/probe shards from a deterministic fold cycle."""
        generator = random.Random(args.seed + salt)
        order = list(range(len(rows)))
        generator.shuffle(order)
        probe_size = args.crossfit_dual_probe_size
        fold_count = math.ceil(len(rows) / probe_size)
        fold = (step_number - 1) % fold_count
        begin = fold * probe_size
        probe_indices = order[begin:begin + probe_size]
        if len(probe_indices) < probe_size:
            probe_indices += order[:probe_size - len(probe_indices)]
        probe_set = set(probe_indices)
        train_indices = [index for index in order if index not in probe_set]
        return (
            [rows[index] for index in train_indices[rank::world]],
            [rows[index] for index in probe_indices[rank::world]],
        )

    def fixed_heldout_rows_for_step(
        rows, step_number, salt, probe_size, primal_size,
    ):
        """Sample a fresh primal batch while keeping a fixed probe disjoint."""
        split_rng = random.Random(args.seed + salt)
        order = list(range(len(rows)))
        split_rng.shuffle(order)
        probe_indices = order[:probe_size]
        primal_pool = order[probe_size:]
        sample_rng = random.Random(
            args.seed + salt + 32452843 * step_number
        )
        if primal_size <= len(primal_pool):
            primal_indices = sample_rng.sample(primal_pool, primal_size)
        else:
            primal_indices = [
                primal_pool[sample_rng.randrange(len(primal_pool))]
                for _ in range(primal_size)
            ]
        return (
            [rows[index] for index in primal_indices[rank::world]],
            [rows[index] for index in probe_indices[rank::world]],
        )
    replay_schedule = build_replay_schedule(
        replay_pools,
        steps if (args.joint_projection or args.fast_only) else 2 * steps,
        args.seed,
        mode=args.replay_mode,
        rank=rank,
        world=world,
        sample_size=args.replay_sample_size,
    ) if replay_pools else []
    fixed_fast_scale = {"values": None}
    calibration_history: list[torch.Tensor] = []

    start_time = time.monotonic()
    if rank == 0:
        print(json.dumps({
            "event": "config",
            "method": "LAPS-Simplex-STCH",
            "task": args.task,
            "stage": args.stage,
            "task_count": args.stage + 1,
            "train_rows": len(current_rows),
            "rejected_rows": rejected,
            "replay_rows": len(replay_rows),
            "replay_overlength": args.replay_overlength,
            "replay_truncated_rows": sum(
                bool(row.get("prompt_truncated", False))
                for row in replay_rows
            ),
            "replay_rows_by_task": {
                TASKS[key]: len(value) for key, value in replay_pools.items()
            },
            "steps_per_phase": steps,
            "optimization_steps": (
                steps + args.current_apex_warmup_steps if args.fast_only else
                steps if (
                    args.joint_projection
                    or args.acquire_only_current_apex
                ) else 2 * steps
            ),
            "current_apex_warmup_steps": args.current_apex_warmup_steps,
            "fast_calibration_steps": (
                fast_only_calibration_steps if args.fast_only else
                args.fast_calibration_steps if args.joint_projection else 0
            ),
            "fast_scale_statistic": "mean of calibration steps 6-10",
            "epochs": epochs,
            "global_current_batch": current_global_batch,
            "crossfit_dual_probe_size": args.crossfit_dual_probe_size,
            "heldout_current_dual_probe_size": (
                args.heldout_current_dual_probe_size
            ),
            "heldout_replay_dual_probe_size": (
                args.heldout_replay_dual_probe_size
            ),
            "crossfit_primal_current_rows": (
                len(current_rows) - args.crossfit_dual_probe_size
                if args.crossfit_dual_probe_size else None
            ),
            "crossfit_primal_replay_rows": (
                len(replay_rows) - args.crossfit_dual_probe_size
                if args.crossfit_dual_probe_size else None
            ),
            "current_sampling": (
                "full eligible train set; independent global sampling with replacement; "
                "batch cardinality matched to replay buffer"
                if args.current_with_replacement and args.current_batch_match_replay else
                "full eligible train set; independent global sampling with replacement"
                if args.current_with_replacement else
                "full eligible train set; global without-replacement shuffle-cycle; "
                "batch cardinality matched to replay buffer"
                if args.current_batch_match_replay else
                "configured global batch over shuffled training rows"
            ),
            "global_replay_batch": (
                0 if args.stage == 0 else
                len(replay_rows) if args.replay_mode == "full_buffer" else
                args.replay_sample_size if args.replay_mode == "sample_fixed" else
                args.stage
            ),
            "global_replay_rows_per_old_task": (
                0 if args.stage == 0 else
                "all" if args.replay_mode == "full_buffer" else 1
                if args.replay_mode == "per_task_one" else
                "proportional_stratified"
            ),
            "replay_sampling": (
                "entire replay buffer every step; deterministic rank sharding"
                if args.replay_mode == "full_buffer" else
                "fixed-size proportional stratified subset; without-replacement shuffle-cycle"
                if args.replay_mode == "sample_fixed" else
                "global one per old task; without-replacement shuffle-cycle"
            ),
            "world_size": world,
            "per_rank_current_batch": (
                [local_batch_min, local_batch_max]
                if uneven_current_sharding else local_batch
            ),
            "preference_sampling": (
                (
                    "recursive stratified: new apex + old face + two edges + "
                    f"{args.preference_count - 4} Dirichlet({args.preference_dirichlet_alpha:g}) interior draws"
                    if args.recursive_stratified_preferences else
                    f"1 cyclic simplex vertex + {args.preference_count - 1} "
                    f"Dirichlet({args.preference_dirichlet_alpha:g}) draws"
                    if args.include_one_cyclic_vertex else
                    f"{args.preference_count} full-simplex Dirichlet({args.preference_dirichlet_alpha:g}) draws; no forced vertices"
                )
                if args.fast_only else
                f"{args.preference_count} full-simplex Dirichlet({args.preference_dirichlet_alpha:g}) draws"
                if args.joint_projection else
                f"phase_A uniform barycenter; phase_B Dirichlet({args.preference_dirichlet_alpha:g})"
            ),
            "fast_control_initialization": (
                "fresh base-embedding initialization"
                if args.fresh_fast_controls else "incremental previous prompt"
            ),
            "slow_objective": (
                "current-task-only SFT jointly conditioned on the new task apex"
                if args.acquire_only_current_apex else
                "frozen SFT+Replay checkpoint"
                if args.fast_only else
                "mean current gradient projected onto mean old-task replay-safe half-space"
                if args.joint_projection else
                "phase_A uniform-barycenter STCH"
            ),
            "fast_objective": (
                "new task vertex only; inherited old face frozen"
                if args.acquire_only_current_apex else
                f"mean {args.scalarization} over {args.preference_count} random full-simplex preferences"
                if args.fast_only else
                "post-calibration fixed-scale STCH over four random preferences"
                if args.joint_projection else
                "phase_A barycenter STCH + phase_B mean STCH over four preferences"
            ),
            "acquire_objective": (
                "per-example relative Softplus regret against frozen Slow"
                if args.relative_softplus_tau > 0 else
                "current-task gold-response SFT NLL"
            ),
            "preserve_objective": (
                "gold-response replay SFT NLL"
                if args.preserve_objective == "sft" else
                "old-checkpoint top-64+OTHER teacher-forcing FKL"
            ),
            "normalization": (
                "disabled (raw task losses)"
                if (
                    args.scalarization == "weighted_sum"
                    or args.disable_stch_normalization
                ) else
                "fixed per-task scales from mean of calibration steps 6-10"
                if args.fast_only and fast_only_calibration_steps > 0 else
                "single-objective; no normalization"
                if args.fast_only else
                "fixed s_P/s_A from mean of calibration steps 6-10"
                if args.joint_projection else
                "cumulative min/max from detached losses; "
                "ideal=0.9*running_min; range=running_max-ideal"
            ),
            "fast_fixed_scales": (
                fixed_fast_scale["values"].detach().cpu().tolist()
                if fixed_fast_scale["values"] is not None else None
            ),
            "relative_softplus_tau": args.relative_softplus_tau,
            "batch_regret_margin": args.batch_regret_margin,
            "batch_regret_tau": args.batch_regret_tau,
            "batch_regret_weight": args.batch_regret_weight,
            "taskwise_regret_margin": args.taskwise_regret_margin,
            "taskwise_regret_tau": args.taskwise_regret_tau,
            "taskwise_dual_lr": args.taskwise_dual_lr,
            "taskwise_dual_init": args.taskwise_dual_init,
            "taskwise_dual_ema_decay": args.taskwise_dual_ema_decay,
            "apex_specific_taskwise_dual": bool(
                args.apex_specific_taskwise_dual
            ),
            "old_face_controls_frozen": bool(
                args.stage > 0 and not args.joint_projection
                and not args.train_all_fast_controls
            ),
            "new_control_count": int((~old_face_mask).sum()),
            "all_vertices_frozen_during_fast": bool(args.freeze_all_vertices),
            "fast_trainable_control_count": int((~fast_frozen_mask).sum()),
            "soft_prompt_placement": "chat_start",
            "preference_count": args.preference_count,
            "preference_chunk": args.preference_chunk,
            "recursive_verification": getattr(
                prompt_model, "recursive_verification", None,
            ),
            **prompt_model.metadata(),
        }), flush=True)

    barycenter = torch.full(
        (args.stage + 1,), 1.0 / (args.stage + 1), device=device,
    )

    # Extra semantic initialization; it does not consume any of the requested
    # full preference-training epochs below.
    if args.fast_only and args.current_apex_warmup_steps:
        for warmup_step in range(1, args.current_apex_warmup_steps + 1):
            if fixed_heldout_dual:
                current_batch, _ = fixed_heldout_rows_for_step(
                    current_rows, warmup_step, 600011,
                    args.heldout_current_dual_probe_size,
                    current_global_batch,
                )
            else:
                indices = current_indices_for_step(warmup_step)
                current_batch = [current_rows[index] for index in indices]
            current_tensors = batch_tensors(current_batch, tokenizer, device)
            metrics = fast_only_current_apex_warmup_step(
                model, prompt_model, current_tensors,
                fast_optimizer, fast_scheduler, args,
            )
            if rank == 0:
                print(json.dumps({
                    "event": "step", "task": args.task, "stage": args.stage,
                    "phase": metrics["phase"], "step": warmup_step,
                    "steps": args.current_apex_warmup_steps,
                    "optimization_step": warmup_step,
                    "optimization_steps": steps + args.current_apex_warmup_steps,
                    "effective_current_epoch": (
                        warmup_step * current_global_batch / len(current_rows)
                    ),
                    "preferences": [[
                        0.0 if index != args.stage else 1.0
                        for index in range(args.stage + 1)
                    ]],
                    "current_rows_per_rank": len(current_batch),
                    "replay_rows_per_rank": 0,
                    "completion_length_mean": float(
                        current_tensors["target_mask"].sum(-1).float().mean()
                    ),
                    "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "elapsed_s": time.monotonic() - start_time,
                    **metrics,
                }), flush=True)
            torch.cuda.reset_peak_memory_stats(device)

        # Start the full-simplex objective from clean Adam moments and a fresh
        # schedule; otherwise the endpoint phase consumes most of the useful LR.
        fast_optimizer = make_fast_optimizer()
        main_fast_schedule_steps = max(1, steps - fast_only_calibration_steps)
        fast_scheduler = make_scheduler(
            fast_optimizer, main_fast_schedule_steps,
            0,
            args.lr_scheduler_type,
        )
        if rank == 0:
            print(json.dumps({
                "event": "fast_optimizer_reset_after_apex_warmup",
                "task": args.task,
                "stage": args.stage,
                "main_steps": main_fast_schedule_steps,
                "fast_lr": args.fast_lr,
                "lr_scheduler_type": args.lr_scheduler_type,
            }), flush=True)

    if args.joint_projection:
        for step in range(1, steps + 1):
            indices = current_indices_for_step(step)
            current_batch = [current_rows[index] for index in indices]
            replay_batch = replay_schedule[step - 1] if replay_schedule else []
            current_tensors = batch_tensors(current_batch, tokenizer, device)
            replay_tensors = (
                batch_tensors(replay_batch, tokenizer, device)
                if replay_batch else None
            )
            preserve_targets = (
                preserve_teacher_targets(
                    preserve_teacher, old_prompt_model, replay_tensors,
                    args.preserve_top_k, args.condition_chunk,
                ) if (
                    replay_tensors is not None
                    and args.preserve_objective == "topk_fkl"
                ) else None
            )
            preferences = (
                sample_recursive_preferences(
                    args.stage + 1, step, args.seed, device,
                    count=args.preference_count,
                    alpha=args.preference_dirichlet_alpha,
                ) if args.recursive_stratified_preferences else
                sample_random_preferences(
                    args.stage + 1, step, args.seed, device,
                    count=args.preference_count,
                    alpha=args.preference_dirichlet_alpha,
                    include_one_cyclic_vertex=args.include_one_cyclic_vertex,
                )
            )
            update_fast = step > args.fast_calibration_steps
            diagnose_fast = update_fast and (
                step == args.fast_calibration_steps + 1 or step % 10 == 0
            )
            metrics = joint_projection_step(
                model, prompt_model, current_tensors, replay_tensors,
                preserve_targets, preferences,
                slow_optimizer, fast_optimizer, slow_scheduler, fast_scheduler,
                fixed_fast_scale, update_fast, diagnose_fast, args,
            )
            calibration_history.append(torch.tensor([
                metrics["calibration_batch_preserve"],
                metrics["calibration_batch_acquire"],
            ], device=device))
            if step == args.fast_calibration_steps:
                tail = torch.stack(calibration_history[-5:])
                calibrated = tail.mean(dim=0).clamp_min(1e-6)
                fixed_fast_scale["values"] = torch.cat((
                    calibrated[0].expand(args.stage),
                    calibrated[1].unsqueeze(0),
                )).detach()
                metrics["calibration_finalized"] = True
                metrics["calibrated_fast_scales"] = (
                    fixed_fast_scale["values"].cpu().tolist()
                )
            else:
                metrics["calibration_finalized"] = False
            if rank == 0:
                current_controls = prompt_model.controls.detach().float()
                old_face_delta = (
                    current_controls[old_face_mask] - initial_controls[old_face_mask]
                    if torch.any(old_face_mask) else
                    current_controls.new_zeros(1)
                )
                print(json.dumps({
                    "event": "step",
                    "task": args.task,
                    "stage": args.stage,
                    "phase": metrics["phase"],
                    "step": step,
                    "steps": steps,
                    "optimization_step": step,
                    "optimization_steps": steps,
                    "effective_current_epoch": (
                        step * current_global_batch / len(current_rows)
                    ),
                    "preferences": preferences.detach().cpu().tolist(),
                    "current_zero_preference_count": 0,
                    "current_rows_per_rank": len(current_batch),
                    "replay_rows_per_rank": len(replay_batch),
                    "completion_length_mean": float(
                        current_tensors["target_mask"].sum(-1).float().mean()
                    ),
                    "peak_memory_gib": (
                        torch.cuda.max_memory_allocated(device) / 2**30
                    ),
                    "elapsed_s": time.monotonic() - start_time,
                    "old_face_control_drift_rms": float(
                        old_face_delta.pow(2).mean().sqrt()
                    ),
                    "old_face_control_drift_norm": float(old_face_delta.norm()),
                    **metrics,
                }), flush=True)
            torch.cuda.reset_peak_memory_stats(device)

    phase_steps = 0 if (args.joint_projection or args.fast_only) else steps

    # Phase A: one barycenter forward jointly updates theta and new controls.
    for step in range(1, phase_steps + 1):
        indices = current_indices_for_step(step)
        current_batch = [current_rows[index] for index in indices]
        replay_batch = (
            [] if args.acquire_only_current_apex else
            replay_schedule[step - 1] if replay_schedule else []
        )
        current_completion_length_mean = (
            sum(len(row["answer_ids"]) for row in current_batch)
            / len(current_batch)
        )
        if args.acquire_only_current_apex:
            current_tensors = batch_tensors(current_batch, tokenizer, device)
            replay_tensors = None
            condition_batches_merged = False
        else:
            current_tensors, replay_tensors, condition_batches_merged = (
                fast_condition_tensors(
                    current_batch, replay_batch, tokenizer, device,
                    args.preserve_objective,
                )
            )
        preserve_targets = (
            preserve_teacher_targets(
                preserve_teacher, old_prompt_model, replay_tensors,
                args.preserve_top_k, args.condition_chunk,
            ) if (
                replay_tensors is not None
                and args.preserve_objective == "topk_fkl"
            ) else None
        )
        if args.acquire_only_current_apex:
            metrics = current_apex_acquire_step(
                model, prompt_model, current_tensors,
                slow_optimizer, fast_optimizer, slow_scheduler, fast_scheduler,
                old_face_mask, args,
            )
            active_preference = torch.zeros_like(barycenter)
            active_preference[-1] = 1.0
        else:
            metrics = phase_a_step(
                model, prompt_model, current_tensors, replay_tensors,
                preserve_targets, barycenter,
                slow_optimizer, fast_optimizer, slow_scheduler, fast_scheduler,
                normalization, old_face_mask, args,
            )
            active_preference = barycenter
        if rank == 0:
            current_controls = prompt_model.controls.detach().float()
            old_face_delta = (
                current_controls[old_face_mask] - initial_controls[old_face_mask]
                if torch.any(old_face_mask) else
                current_controls.new_zeros(1)
            )
            print(json.dumps({
                "event": "step",
                "task": args.task,
                "stage": args.stage,
                "phase": metrics["phase"],
                "step": step,
                "steps": steps,
                "optimization_step": step,
                "optimization_steps": (
                    steps if args.acquire_only_current_apex else 2 * steps
                ),
                "effective_current_epoch": (
                    step * current_global_batch / len(current_rows)
                ),
                "preferences": [active_preference.detach().cpu().tolist()],
                "uniform_preference": (
                    None if args.acquire_only_current_apex else
                    barycenter.detach().cpu().tolist()
                ),
                "current_rows_per_rank": len(current_batch),
                "replay_rows_per_rank": len(replay_batch),
                "condition_rows_per_rank": len(current_tensors["task_ids"]),
                "condition_batches_merged": condition_batches_merged,
                "completion_length_mean": current_completion_length_mean,
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "elapsed_s": time.monotonic() - start_time,
                "old_face_control_drift_rms": float(
                    old_face_delta.pow(2).mean().sqrt()
                ),
                "old_face_control_drift_norm": float(old_face_delta.norm()),
                **metrics,
            }), flush=True)
        torch.cuda.reset_peak_memory_stats(device)

    # Phase B traverses the same shuffled task sequence. Theta is frozen. In
    # fast-only mode it is the sole phase and all controls may realign to the
    # newly loaded SFT+Replay backbone.
    fast_phase_steps = (
        0 if args.acquire_only_current_apex else
        steps if args.fast_only else phase_steps
    )
    fast_only_scale_history: list[torch.Tensor] = []
    fast_only_fixed_scales = None
    for step in range(1, fast_phase_steps + 1):
        current_probe_batch = []
        replay_probe_batch = []
        if args.crossfit_dual_probe_size:
            current_batch, current_probe_batch = crossfit_rows_for_step(
                current_rows, step, 600011,
            )
            replay_batch, replay_probe_batch = crossfit_rows_for_step(
                replay_rows, step, 700001,
            )
        elif fixed_heldout_dual:
            current_batch, current_probe_batch = fixed_heldout_rows_for_step(
                current_rows, step, 600011,
                args.heldout_current_dual_probe_size,
                current_global_batch,
            )
            replay_batch, replay_probe_batch = fixed_heldout_rows_for_step(
                replay_rows, step, 700001,
                args.heldout_replay_dual_probe_size,
                args.replay_sample_size,
            )
        else:
            indices = current_indices_for_step(step)
            current_batch = [current_rows[index] for index in indices]
            replay_offset = 0 if args.fast_only else steps
            replay_batch = replay_schedule[
                replay_offset + step - 1
            ] if replay_schedule else []
        current_completion_length_mean = (
            sum(len(row["answer_ids"]) for row in current_batch)
            / len(current_batch)
        )
        current_tensors, replay_tensors, condition_batches_merged = (
            fast_condition_tensors(
                current_batch, replay_batch, tokenizer, device,
                args.preserve_objective,
            )
        )
        dual_probe_current_tensors = None
        dual_probe_replay_tensors = None
        if args.crossfit_dual_probe_size or fixed_heldout_dual:
            (
                dual_probe_current_tensors,
                dual_probe_replay_tensors,
                _,
            ) = fast_condition_tensors(
                current_probe_batch, replay_probe_batch, tokenizer, device,
                args.preserve_objective,
            )
        if args.relative_softplus_tau > 0:
            attach_slow_reference_losses(
                model, current_tensors, args.condition_chunk,
            )
            attach_slow_reference_losses(
                model, replay_tensors, args.condition_chunk,
            )
            attach_slow_reference_losses(
                model, dual_probe_current_tensors, args.condition_chunk,
            )
            attach_slow_reference_losses(
                model, dual_probe_replay_tensors, args.condition_chunk,
            )
        condition_chunk_sizes = [
            end - begin
            for begin, end in adaptive_row_ranges(
                current_tensors, args.condition_chunk,
                args.preference_count, args.prompt_length,
                args.condition_token_budget,
            )
        ]
        preserve_targets = (
            preserve_teacher_targets(
                preserve_teacher, old_prompt_model, replay_tensors,
                args.preserve_top_k, args.condition_chunk,
            ) if (
                replay_tensors is not None
                and args.preserve_objective == "topk_fkl"
            ) else None
        )
        preferences = (
            sample_recursive_preferences(
                args.stage + 1, step, args.seed, device,
                count=args.preference_count,
                alpha=args.preference_dirichlet_alpha,
            ) if args.recursive_stratified_preferences else
            sample_random_preferences(
                args.stage + 1, step, args.seed, device,
                count=args.preference_count,
                alpha=args.preference_dirichlet_alpha,
                include_one_cyclic_vertex=args.include_one_cyclic_vertex,
            )
        )
        if args.fast_only and step <= fast_only_calibration_steps:
            metrics = fast_only_calibration_step(
                model, prompt_model, current_tensors, replay_tensors,
                preserve_targets, preferences, args,
            )
            fast_only_scale_history.append(torch.tensor(
                metrics["calibration_task_means"], device=device,
            ))
            if step == fast_only_calibration_steps:
                # Use the stable tail of the initial calibration window.  For
                # the default ten-step window this is exactly steps 6--10.
                tail_size = min(5, len(fast_only_scale_history))
                fast_only_fixed_scales = torch.stack(
                    fast_only_scale_history[-tail_size:],
                ).mean(0).clamp_min(1e-6)
                metrics["calibration_finalized"] = True
                metrics["fast_fixed_scales"] = (
                    fast_only_fixed_scales.detach().cpu().tolist()
                )
            else:
                metrics["calibration_finalized"] = False
        else:
            metrics = phase_b_step(
                model, prompt_model, current_tensors, replay_tensors,
                preserve_targets, preferences,
                fast_optimizer, fast_scheduler, normalization,
                fast_frozen_mask, args,
                fixed_scales=(
                    None if args.disable_stch_normalization
                    else fast_only_fixed_scales
                ),
                taskwise_dual=taskwise_dual,
                taskwise_violation_ema=taskwise_violation_ema,
                dual_probe_current_tensors=dual_probe_current_tensors,
                dual_probe_replay_tensors=dual_probe_replay_tensors,
            )
            metrics["calibration_finalized"] = False
        if rank == 0:
            current_controls = prompt_model.controls.detach().float()
            old_face_delta = (
                current_controls[old_face_mask] - initial_controls[old_face_mask]
                if torch.any(old_face_mask) else
                current_controls.new_zeros(1)
            )
            print(json.dumps({
                "event": "step",
                "phase": "B_fast_random_simplex",
                "task": args.task,
                "stage": args.stage,
                "step": step,
                "steps": steps,
                "optimization_step": (
                    args.current_apex_warmup_steps + step
                    if args.fast_only else steps + step
                ),
                "optimization_steps": (
                    args.current_apex_warmup_steps + steps
                    if args.fast_only else 2 * steps
                ),
                "effective_current_epoch": (
                    step * current_global_batch / len(current_rows)
                ),
                "preferences": preferences.detach().cpu().tolist(),
                "current_rows_per_rank": len(current_batch),
                "replay_rows_per_rank": len(replay_batch),
                "dual_probe_current_rows_per_rank": len(current_probe_batch),
                "dual_probe_replay_rows_per_rank": len(replay_probe_batch),
                "condition_rows_per_rank": len(current_tensors["task_ids"]),
                "condition_batches_merged": condition_batches_merged,
                "condition_chunk_sizes": condition_chunk_sizes,
                "completion_length_mean": current_completion_length_mean,
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "elapsed_s": time.monotonic() - start_time,
                "old_face_control_drift_rms": float(
                    old_face_delta.pow(2).mean().sqrt()
                ),
                "old_face_control_drift_norm": float(old_face_delta.norm()),
                **metrics,
            }), flush=True)
        torch.cuda.reset_peak_memory_stats(device)

    accelerator.wait_for_everyone()
    if rank == 0:
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        if args.hidden_residual:
            torch.save(
                prompt_model.generator.state_dict(),
                args.output_dir / (
                    "layer_simplex_residual_memory.pt"
                    if args.layer_specific_controls else
                    "simplex_residual_memory.pt"
                ),
            )
            torch.save(
                prompt_model.reader.state_dict(),
                args.output_dir / "memory_reader.pt",
            )
        else:
            torch.save(
                prompt_model.state_dict(),
                args.output_dir / "simplex_soft_prompt.pt",
            )
        (args.output_dir / "simplex_config.json").write_text(json.dumps({
            "method": "LAPS-Simplex-STCH",
            "task": args.task,
            "stage": args.stage,
            "task_names": list(TASKS[:args.stage + 1]),
            "preference_semantics": "one coordinate per seen task",
            "current_slow_anchor": bool(args.current_slow_anchor),
            "current_anchor_task": (
                args.task if args.current_slow_anchor else None
            ),
            "current_anchor_exact_identity": bool(args.current_slow_anchor),
            "two_task_scalar_semantics": (
                "lambda=0 is current-task vertex; lambda=1 is old-task vertex"
                if args.stage == 1 else None
            ),
            "slow_preference": [1.0 / (args.stage + 1)] * (args.stage + 1),
            "training_schedule": (
                "current_only_joint_slow_new_apex_acquisition"
                if args.acquire_only_current_apex else
                "fast_only_random_simplex_on_frozen_sft_replay"
                if args.fast_only else
                "10-step slow-only calibration then joint projection/fixed-scale STCH"
                if args.joint_projection else
                "phase_A_joint_barycenter_then_phase_B_fast_random"
            ),
            "old_face_controls_frozen": bool(
                args.stage > 0 and not args.joint_projection
                and not args.train_all_fast_controls
            ),
            "slow_frozen": bool(args.fast_only),
            "hidden_residual": bool(args.hidden_residual),
            "residual_equation": (
                "h_l_prime=h_l+(1-lambda_current)*R_phi_l(h_l,lambda)"
                if args.hidden_residual and args.current_slow_anchor else
                "h_l_prime=h_l+a*R_phi_l(h_l,lambda); a in [0,1]"
                if args.hidden_residual else None
            ),
            "alpha_used_during_training": 1.0 if args.hidden_residual else None,
            "slow_model_checkpoint": str(args.model),
            "preserve_teacher_checkpoint": args.preserve_teacher_model,
            "acquire_objective": (
                "per-example relative Softplus regret against frozen Slow"
                if args.relative_softplus_tau > 0 else
                "current-task gold-response SFT NLL"
            ),
            "preserve_objective": (
                "gold-response replay SFT NLL"
                if args.preserve_objective == "sft" else
                "old-checkpoint top-64+OTHER teacher-forcing FKL"
            ),
            "replay_sampling": (
                "entire replay buffer every step; deterministic rank sharding"
                if args.replay_mode == "full_buffer" else
                "one global row per old task, shuffle-cycle"
            ),
            "normalization": (
                "fixed per-task scales from mean of calibration steps 6-10"
                if args.fast_only and fast_only_calibration_steps > 0 else
                "single-objective; no normalization"
                if args.fast_only else
                "fixed s_P/s_A from mean of calibration steps 6-10"
                if args.joint_projection else
                "cumulative min/max from detached losses; "
                "ideal=0.9*running_min; range=running_max-ideal"
            ),
            "fast_fixed_scales": (
                fast_only_fixed_scales.detach().cpu().tolist()
                if args.fast_only and fast_only_fixed_scales is not None else
                fixed_fast_scale["values"].detach().cpu().tolist()
                if fixed_fast_scale["values"] is not None else None
            ),
            "soft_prompt_placement": "chat_start",
            "recursive_verification": getattr(
                prompt_model, "recursive_verification", None,
            ),
            **prompt_model.metadata(),
        }, indent=2) + "\n")
        if args.hidden_residual:
            (args.output_dir / "method_config.json").write_text(json.dumps({
                "method": "LAPS-zero-preserving-hidden-residual",
                "base_model": str(args.model),
                "task": args.task,
                "stage": args.stage,
                "task_names": list(TASKS[:args.stage + 1]),
                "num_tasks": args.stage + 1,
                "bezier_degree": args.bezier_degree,
                "memory_tokens": args.prompt_length,
                "hidden_size": int(model.config.hidden_size),
                "reader_layers": list(prompt_model.reader.layer_indices),
                "reader_o_lora_rank": args.reader_o_lora_rank,
                "reader_scale_lr": (
                    args.reader_scale_lr
                    if args.reader_scale_lr is not None else args.fast_lr
                ),
                "normalize_reader_residual": False,
                "layer_specific_controls": bool(args.layer_specific_controls),
                "encoder_memory_tokens": (
                    args.prompt_length * len(prompt_model.reader.layer_indices)
                    if args.layer_specific_controls else args.prompt_length
                ),
                "request_specific_alpha_last_dim": False,
                "trainable_layer_alpha": bool(args.trainable_layer_alpha),
                "layer_alphas": (
                    prompt_model.layer_alphas().detach().cpu().tolist()
                    if prompt_model.layer_alphas() is not None else None
                ),
                "reader_scales": prompt_model.reader.layer_scales.detach().cpu().tolist(),
                "bo_search_alpha": False,
                "alpha_semantics": "none; amplitude is learned only through reader s_l",
                "preserve_objective": (
                    "gold replay SFT" if args.preserve_objective == "sft" else
                    "old slow checkpoint top-64+OTHER teacher-forcing FKL"
                ),
            }, indent=2) + "\n")
        (args.output_dir / "STAGE_COMPLETE").touch()
        print(json.dumps({
            "event": "stage_complete",
            "task": args.task,
            "stage": args.stage,
            "output": str(args.output_dir),
            "elapsed_s": time.monotonic() - start_time,
        }), flush=True)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
