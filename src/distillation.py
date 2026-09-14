"""Compressed top-k + OTHER distillation targets used by LAPS."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def teacher_topk_tail_targets(
    teacher_logits: torch.Tensor,
    top_k: int = 64,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress teacher logits into top-k log-probabilities and tail mass."""
    k = min(top_k, teacher_logits.size(-1) - 1)
    log_probs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    top_logp, top_idx = torch.topk(log_probs, k=k, dim=-1)
    top_mass = top_logp.exp().sum(-1).clamp(max=1.0 - 1e-7)
    return top_idx.detach(), top_logp.detach(), top_mass.detach()


def topk_tail_forward_kl_from_targets(
    teacher_top_idx: torch.Tensor,
    teacher_top_logp: torch.Tensor,
    teacher_mass: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Teacher||Student KL over teacher top-k plus one OTHER bin."""
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    student_top_logp = student_logp.gather(-1, teacher_top_idx)
    teacher_top_p = teacher_top_logp.exp()
    student_mass = student_top_logp.exp().sum(-1).clamp(max=1.0 - 1e-7)
    top_kl = (
        teacher_top_p * (teacher_top_logp - student_top_logp)
    ).sum(-1)
    teacher_tail = (1.0 - teacher_mass).clamp_min(1e-7)
    student_tail = (1.0 - student_mass).clamp_min(1e-7)
    tail_kl = teacher_tail * (teacher_tail.log() - student_tail.log())
    return (top_kl + tail_kl) * temperature**2, teacher_mass, student_mass
