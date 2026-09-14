#!/usr/bin/env python3
"""Build a deterministic task-specific calibration buffer for simplex BO."""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", default="NumGLUE-cm")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", default="2026:NumGLUE-cm:train50")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = args.data_root / args.task / "train.json"
    rows = json.loads(source_path.read_text())
    if not 0 < args.size <= len(rows):
        raise ValueError(f"size must be in [1, {len(rows)}], got {args.size}")

    indices = random.Random(args.seed).sample(range(len(rows)), args.size)
    selected = [{**rows[index], "source_index": index} for index in indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    buffer_path = args.output_dir / "buffer.jsonl"
    buffer_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    )
    manifest = {
        "method": "fixed-random-train-calibration",
        "source": str(source_path.resolve()),
        "buffer_size": args.size,
        "allocation": {args.task: args.size},
        "seed": args.seed,
        "source_indices": indices,
    }
    (args.output_dir / "buffer.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(buffer_path)


if __name__ == "__main__":
    main()
