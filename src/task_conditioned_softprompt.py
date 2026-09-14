"""Oracle-task-conditioned cubic Bezier soft-prompt memory."""

from __future__ import annotations

import torch
import torch.nn as nn


class TaskConditionedBezierCode(nn.Module):
    """One cubic Bezier prompt path per oracle task.

    ``controls[t]`` has shape ``[4, prompt_length, hidden_size]``.  The
    parameter count is deliberately small and, unlike an x-conditioned
    hypernetwork, cannot memorize individual replay questions.
    """

    def __init__(
        self, num_tasks: int, prompt_length: int, hidden_size: int,
    ) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.prompt_length = int(prompt_length)
        self.hidden_size = int(hidden_size)
        self.controls = nn.Parameter(torch.zeros(
            self.num_tasks, 4, self.prompt_length, self.hidden_size,
        ))

    @staticmethod
    def basis(lambdas: torch.Tensor) -> torch.Tensor:
        one = 1.0 - lambdas
        return torch.stack((
            one.pow(3),
            3.0 * lambdas * one.pow(2),
            3.0 * lambdas.pow(2) * one,
            lambdas.pow(3),
        ), dim=-1)

    def forward(
        self, lambdas: torch.Tensor, task_ids: torch.Tensor,
    ) -> torch.Tensor:
        selected = self.controls[task_ids.long()]
        basis = self.basis(lambdas.float()).to(selected.dtype)
        return torch.einsum("bk,bkmh->bmh", basis, selected)
