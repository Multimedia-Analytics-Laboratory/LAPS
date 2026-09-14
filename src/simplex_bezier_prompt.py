"""Incremental Bezier soft prompts over a task-preference simplex."""

from __future__ import annotations

import math
from itertools import product

import torch
from torch import nn


def simplex_multi_indices(num_tasks: int, degree: int) -> list[tuple[int, ...]]:
    """All non-negative multi-indices of length ``num_tasks`` summing to degree."""
    if num_tasks < 1 or degree < 0:
        raise ValueError("num_tasks must be positive and degree non-negative")
    return [
        tuple(values)
        for values in product(range(degree + 1), repeat=num_tasks)
        if sum(values) == degree
    ]


class SimplexBezierPrompt(nn.Module):
    """A degree-d Bezier simplex whose control points are soft prompts.

    For preference ``lambda`` on the task simplex,

        Z(lambda) = sum_alpha multinomial(d; alpha) lambda**alpha P_alpha.

    Increasing the task count embeds the previous simplex as the face whose
    newest coordinate is zero.  All controls remain trainable after expansion.
    """

    def __init__(
        self,
        num_tasks: int,
        degree: int,
        prompt_length: int,
        hidden_size: int,
        *,
        init_prompt: torch.Tensor | None = None,
        random_init: bool = False,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.degree = int(degree)
        self.prompt_length = int(prompt_length)
        self.hidden_size = int(hidden_size)
        self.residual_scale = float(residual_scale)
        indices = simplex_multi_indices(self.num_tasks, self.degree)
        self.register_buffer(
            "multi_indices",
            torch.tensor(indices, dtype=torch.long),
            persistent=True,
        )
        controls = torch.zeros(
            len(indices), self.prompt_length, self.hidden_size,
            dtype=torch.float32,
        )
        if random_init:
            # Match torch.nn.Embedding.reset_parameters(): every control point
            # and every prompt position receives an independent N(0, 1) row.
            nn.init.normal_(controls)
        elif init_prompt is not None:
            if tuple(init_prompt.shape) != (self.prompt_length, self.hidden_size):
                raise ValueError(
                    f"init_prompt has shape {tuple(init_prompt.shape)}, expected "
                    f"{(self.prompt_length, self.hidden_size)}"
                )
            controls.copy_(init_prompt.detach().float().unsqueeze(0))
        self.controls = nn.Parameter(controls)

        coefficients = []
        numerator = math.factorial(self.degree)
        for alpha in indices:
            denominator = math.prod(math.factorial(value) for value in alpha)
            coefficients.append(numerator / denominator)
        self.register_buffer(
            "multinomial_coefficients",
            torch.tensor(coefficients, dtype=torch.float32),
            persistent=True,
        )

    def basis(self, preferences: torch.Tensor) -> torch.Tensor:
        if preferences.ndim != 2 or preferences.shape[1] != self.num_tasks:
            raise ValueError(
                f"preferences must have shape [B, {self.num_tasks}], got "
                f"{tuple(preferences.shape)}"
            )
        preferences = preferences.float().clamp_min(0)
        sums = preferences.sum(-1, keepdim=True)
        if torch.any(sums <= 0):
            raise ValueError("each preference must have positive mass")
        preferences = preferences / sums
        powers = self.multi_indices.to(preferences.device)
        terms = preferences.unsqueeze(1).pow(powers.unsqueeze(0)).prod(-1)
        return terms * self.multinomial_coefficients.to(preferences.device)

    def forward(
        self,
        preferences: torch.Tensor,
        *,
        isolate_vertex_gradients: bool = False,
    ) -> torch.Tensor:
        basis = self.basis(preferences).to(self.controls.dtype)
        # Pure vertex controls P_{d e_t} are updated only by their exact task
        # apex when isolation is enabled. Interior preferences retain the same
        # numerical forward value, but see detached vertex controls during
        # backward. Mixed controls remain trainable at every preference.
        if isolate_vertex_gradients:
            vertex_task = self.multi_indices.eq(self.degree).float().argmax(-1)
            is_vertex_control = self.multi_indices.eq(self.degree).any(-1)
            exact_vertex = preferences.float().ge(1.0 - 1e-6)
            live = (
                ~is_vertex_control.unsqueeze(0)
                | exact_vertex[:, vertex_task].to(is_vertex_control.device)
            ).to(basis.dtype)
            # Fold the optional centering transform into the linear basis so
            # forward equivalence and gradient isolation both remain exact.
            effective_basis = (
                self.residual_scale * basis
                + (1.0 - self.residual_scale) / basis.shape[-1]
            )
            prompt = torch.einsum(
                "bc,clh->blh", effective_basis * live, self.controls,
            ) + torch.einsum(
                "bc,clh->blh",
                effective_basis * (1.0 - live),
                self.controls.detach(),
            )
        else:
            prompt = torch.einsum("bc,clh->blh", basis, self.controls)
            if self.residual_scale != 1.0:
                center = self.controls.mean(0, keepdim=True)
                prompt = center + self.residual_scale * (prompt - center)
        return prompt

    @classmethod
    def expand_from_state(
        cls,
        state: dict[str, torch.Tensor],
        num_tasks: int,
        *,
        init_prompt: torch.Tensor | None = None,
        random_init: bool = False,
    ) -> "SimplexBezierPrompt":
        old_indices = state["multi_indices"].long()
        old_controls = state["controls"].float()
        old_tasks = int(old_indices.shape[1])
        degree = int(old_indices[0].sum())
        if num_tasks != old_tasks + 1:
            raise ValueError(
                f"incremental expansion requires {old_tasks + 1} tasks, got {num_tasks}"
            )
        model = cls(
            num_tasks, degree, old_controls.shape[1], old_controls.shape[2],
            init_prompt=init_prompt, random_init=random_init,
        )
        lookup = {
            tuple(alpha.tolist()): index
            for index, alpha in enumerate(model.multi_indices)
        }
        with torch.no_grad():
            for alpha, control in zip(old_indices.tolist(), old_controls):
                model.controls[lookup[tuple(alpha) + (0,)]].copy_(control)
        return model

    def metadata(self) -> dict[str, int | float]:
        return {
            "num_tasks": self.num_tasks,
            "degree": self.degree,
            "control_count": int(self.controls.shape[0]),
            "prompt_length": self.prompt_length,
            "hidden_size": self.hidden_size,
            "residual_scale": self.residual_scale,
        }


@torch.no_grad()
def verify_recursive_expansion(
    old: SimplexBezierPrompt,
    expanded: SimplexBezierPrompt,
    *,
    tolerance: float = 2e-6,
) -> dict[str, float]:
    """Verify the exact normalized-recursion semantics of one expansion.

    The direct multivariate Bernstein evaluation does not explicitly divide
    historical coordinates by ``1-alpha``.  It is nevertheless exactly equal
    to the normalized recursive form because every old-face degree-d basis
    term contributes a common factor ``(1-alpha)**d``.  This assertion guards
    both that identity and exact boundary inheritance.
    """
    if expanded.num_tasks != old.num_tasks + 1:
        raise ValueError("expanded prompt must add exactly one task")
    if expanded.degree != old.degree:
        raise ValueError("old and expanded prompts must have equal degree")
    if old.residual_scale != 1.0 or expanded.residual_scale != 1.0:
        raise ValueError(
            "recursive verification currently requires residual_scale=1"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260826 + 97 * old.num_tasks)
    historical = torch.rand(8, old.num_tasks, generator=generator)
    historical /= historical.sum(-1, keepdim=True)
    alpha = torch.tensor(
        [0.0, 0.05, 0.20, 0.50, 0.80, 0.95, 1.0, 0.37],
        dtype=torch.float32,
    ).unsqueeze(-1)
    full = torch.cat((historical * (1.0 - alpha), alpha), dim=-1)

    old_face_mask = expanded.multi_indices[:, -1].eq(0)
    actual_old_component = torch.einsum(
        "bc,clh->blh",
        expanded.basis(full)[:, old_face_mask].to(expanded.controls.dtype),
        expanded.controls[old_face_mask],
    )
    expected_old_component = (
        (1.0 - alpha).pow(old.degree).unsqueeze(-1)
        * old(historical)
    )
    recursive_error = float(
        (actual_old_component - expected_old_component).abs().max()
    )

    boundary = torch.cat(
        (historical, torch.zeros(len(historical), 1)), dim=-1,
    )
    boundary_error = float((expanded(boundary) - old(historical)).abs().max())
    basis_partition_error = float(
        (expanded.basis(full).sum(-1) - 1.0).abs().max()
    )
    maximum = max(recursive_error, boundary_error, basis_partition_error)
    if maximum > tolerance:
        raise RuntimeError(
            "invalid recursive Bezier expansion: "
            f"recursive={recursive_error:.3e}, boundary={boundary_error:.3e}, "
            f"partition={basis_partition_error:.3e}, tolerance={tolerance:.3e}"
        )
    return {
        "recursive_old_component_max_error": recursive_error,
        "recursive_boundary_max_error": boundary_error,
        "bernstein_partition_max_error": basis_partition_error,
    }
