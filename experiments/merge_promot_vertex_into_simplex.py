#!/usr/bin/env python3
"""Expand a historical Bezier simplex and install a learned ProMoT endpoint."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.simplex_bezier_prompt import SimplexBezierPrompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--promot-prompt", type=Path, required=True)
    p.add_argument("--previous-prompt", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    # This script runs as a fresh process at every stage.  Seed the random
    # mixed-control initialization so independently launched rebase and
    # no-rebase runs start from the same expanded simplex.
    torch.manual_seed(args.seed)
    learned = torch.load(args.promot_prompt, map_location="cpu", weights_only=True)["prompt_embeddings"].float()
    if args.stage == 0:
        simplex = SimplexBezierPrompt(1, args.degree, learned.shape[0], learned.shape[1], random_init=True)
    else:
        if args.previous_prompt is None:
            raise ValueError("--previous-prompt is required after stage zero")
        state = torch.load(args.previous_prompt, map_location="cpu", weights_only=True)
        simplex = SimplexBezierPrompt.expand_from_state(state, args.stage + 1, random_init=True)
    vertex = simplex.multi_indices[:, args.stage].eq(args.degree)
    if int(vertex.sum()) != 1:
        raise RuntimeError("current simplex vertex is not unique")
    with torch.no_grad():
        # Boolean advanced indexing returns a copy; assignment is required to
        # mutate the underlying control tensor.
        simplex.controls[vertex] = learned.unsqueeze(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(simplex.state_dict(), args.output)


if __name__ == "__main__":
    main()
