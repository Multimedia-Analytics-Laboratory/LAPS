"""Self-contained preference-conditioned soft-prompt utilities."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def prompt_position(tokenizer, ids: list[int]) -> int:
    """Insert memory immediately before the assistant-message boundary."""
    marker = tokenizer.convert_tokens_to_ids("<|im_start|>")
    positions = [i for i, value in enumerate(ids) if value == marker]
    return positions[1] if len(positions) >= 2 else 0


class CubicBezierCode(nn.Module):
    """Map lambda in [0, 1] to a continuous soft prompt."""

    def __init__(self, prompt_length: int, hidden_size: int) -> None:
        super().__init__()
        self.prompt_length = int(prompt_length)
        self.hidden_size = int(hidden_size)
        self.controls = nn.Parameter(torch.zeros(
            4, self.prompt_length, self.hidden_size,
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

    def forward(self, lambdas: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "bk,kmh->bmh",
            self.basis(lambdas.float()).to(self.controls.dtype),
            self.controls,
        )


def stch_loss(acquire, preserve, lambdas, ideals, ranges, mu):
    """Smooth Tchebycheff scalarization for acquisition/preservation."""
    normalized_a = (acquire - ideals[0]) / ranges[0]
    normalized_p = (preserve - ideals[1]) / ranges[1]
    scores = torch.stack((
        (1.0 - lambdas) * normalized_a / mu,
        lambdas * normalized_p / mu,
    ), dim=-1)
    loss = mu * torch.logsumexp(scores, dim=-1)
    weights = scores.softmax(dim=-1)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
    return loss, weights, entropy


def padded_prompt_ids(tokenizer, prompts_text, max_prompt_length, device):
    values = [
        tokenizer(
            text, add_special_tokens=False, truncation=True,
            max_length=max_prompt_length,
        )["input_ids"]
        for text in prompts_text
    ]
    width = max(map(len, values))
    batch = torch.full(
        (len(values), width), tokenizer.pad_token_id,
        dtype=torch.long, device=device,
    )
    mask = torch.zeros_like(batch)
    positions = torch.empty(len(values), dtype=torch.long, device=device)
    for i, ids in enumerate(values):
        offset = width - len(ids)
        batch[i, offset:] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[i, offset:] = 1
        positions[i] = offset + prompt_position(tokenizer, ids)
    return batch, mask, positions, values


def insert_latents(embedding, input_ids, attention_mask, latents, positions):
    token_embeddings = embedding(input_ids)
    latents = latents.to(token_embeddings.dtype)
    rows, masks = [], []
    for i, position in enumerate(positions.tolist()):
        rows.append(torch.cat((
            token_embeddings[i, :position], latents[i],
            token_embeddings[i, position:],
        )))
        masks.append(torch.cat((
            attention_mask[i, :position],
            attention_mask.new_ones(latents.shape[1]),
            attention_mask[i, position:],
        )))
    return torch.stack(rows), torch.stack(masks)


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Log-softmax values at target token indices without retaining probabilities."""
    if logits.dtype in (torch.float32, torch.float64):
        selected = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        return selected - torch.logsumexp(logits, dim=-1)
    rows = []
    for row_logits, row_index in zip(logits, index):
        row_logps = row_logits.log_softmax(dim=-1)
        rows.append(row_logps.gather(dim=-1, index=row_index.unsqueeze(-1)).squeeze(-1))
    return torch.stack(rows)
