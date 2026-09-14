"""Question-conditioned preference soft prompts.

The frozen BGE encoder supplies a semantic feature for each question.  A
compact low-rank hypernetwork turns that feature and the cubic preference
coordinates into a question-specific 32-token prompt.  A global cubic curve
is retained so this module is a strict extension of ``CubicBezierCode``.
"""

from __future__ import annotations

import os
import math

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"


class InputConditionedBezierCode(nn.Module):
    def __init__(
        self,
        prompt_length: int,
        hidden_size: int,
        condition_width: int = 512,
        condition_rank: int = 8,
        max_question_length: int = 384,
        conditional_gate: bool = True,
        bezier_order: int = 3,
    ) -> None:
        super().__init__()
        self.prompt_length = int(prompt_length)
        self.hidden_size = int(hidden_size)
        self.condition_width = int(condition_width)
        self.condition_rank = int(condition_rank)
        self.max_question_length = int(max_question_length)
        self.conditional_gate = bool(conditional_gate)
        self.bezier_order = int(bezier_order)
        if self.bezier_order < 1:
            raise ValueError("bezier_order must be positive")
        self.control_count = self.bezier_order + 1
        self.encoder_name = os.environ.get("PREFERENCE_ENCODER", DEFAULT_ENCODER)

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.encoder_name, local_files_only=True,
        )
        self.text_encoder = AutoModel.from_pretrained(
            self.encoder_name, torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        encoder_hidden = int(self.text_encoder.config.hidden_size)

        # The first term exactly recovers the preceding global Bezier model.
        self.controls = nn.Parameter(torch.zeros(
            self.control_count, self.prompt_length, self.hidden_size,
        ))

        # The hypernetwork predicts per-token coefficients over a small shared
        # basis rather than emitting prompt_length * hidden_size values.
        self.conditioner = nn.Sequential(
            nn.Linear(encoder_hidden + self.control_count, self.condition_width),
            nn.SiLU(),
            nn.Linear(
                self.condition_width,
                self.prompt_length * self.condition_rank,
            ),
        )
        self.condition_basis = nn.Parameter(torch.empty(
            self.condition_rank, self.hidden_size,
        ))
        nn.init.normal_(self.condition_basis, std=0.02)
        # Start from the exact old model (all prompts zero), while preserving a
        # usable first-step gradient through the final conditioner layer.
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def basis(self, lambdas: torch.Tensor) -> torch.Tensor:
        one = 1.0 - lambdas
        order = self.bezier_order
        return torch.stack([
            math.comb(order, index)
            * one.pow(order - index)
            * lambdas.pow(index)
            for index in range(self.control_count)
        ], dim=-1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    @torch.no_grad()
    def encode_questions(self, questions: list[str], device: torch.device) -> torch.Tensor:
        encoded = self.text_tokenizer(
            questions, padding=True, truncation=True,
            max_length=self.max_question_length, return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        # BGE-small uses its first-token representation.
        features = self.text_encoder(**encoded, return_dict=True).last_hidden_state[:, 0]
        return features.detach()

    def forward(
        self, lambdas: torch.Tensor, question_features: torch.Tensor,
    ) -> torch.Tensor:
        cubic = self.basis(lambdas.float())
        global_prompt = torch.einsum(
            "bk,kmh->bmh", cubic.to(self.controls.dtype), self.controls,
        )
        condition_input = torch.cat((
            question_features.to(cubic.dtype), cubic,
        ), dim=-1).to(self.conditioner[0].weight.dtype)
        coefficients = self.conditioner(condition_input).view(
            -1, self.prompt_length, self.condition_rank,
        )
        conditional = torch.einsum(
            "bmr,rh->bmh", coefficients, self.condition_basis,
        )
        # Historical runs explicitly suppressed the question-conditioned
        # branch at the preserve endpoint.  New runs can disable that
        # asymmetric capacity constraint and let lambda inside the
        # conditioner learn the appropriate strength at every preference.
        if self.conditional_gate:
            conditional = conditional * (1.0 - lambdas).to(conditional.dtype)[:, None, None]
        return global_prompt + conditional

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            key: value for key, value in state.items()
            if not key.startswith("text_encoder.")
        }

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing = [
            key for key in result.missing_keys
            if not key.startswith("text_encoder.")
        ]
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(
                f"Input-conditioned prompt state mismatch: missing={missing}, "
                f"unexpected={list(result.unexpected_keys)}"
            )
        return result
