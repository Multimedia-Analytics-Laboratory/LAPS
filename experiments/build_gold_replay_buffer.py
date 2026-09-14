#!/usr/bin/env python3
"""Build a deterministic fixed-budget vanilla replay buffer from gold data."""

import argparse
import json
import os
import random
from pathlib import Path


_CANONICAL_TASKS = (
    "C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
    "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
)
TASKS = tuple(filter(None, os.environ.get(
    "TRACE_TASK_ORDER", ",".join(_CANONICAL_TASKS),
).split(",")))
if len(TASKS) != 8 or set(TASKS) != set(_CANONICAL_TASKS):
    raise ValueError(f"invalid TRACE_TASK_ORDER: {TASKS}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--buffer-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--include-task", action="store_true",
        help="Retain each example's source task in the output JSONL.",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    seen = args.stage + 1
    base, remainder = divmod(args.buffer_size, seen)
    selected, allocation = [], {}
    for task_id, task in enumerate(TASKS[:seen]):
        train_path = args.data_root / task / "train.jsonl"
        if not train_path.is_file():
            train_path = args.data_root / task / "train.json"
        text = train_path.read_text()
        if train_path.suffix == ".json":
            payload = json.loads(text)
            rows = payload if isinstance(payload, list) else list(payload.values())
        else:
            rows = [
                json.loads(line) for line in text.splitlines() if line.strip()
            ]
        rng = random.Random(args.seed + 1009 * task_id)
        rng.shuffle(rows)
        quota = base + int(task_id < remainder)
        chosen = rows[:quota]
        selected.extend({"task": task, **row} for row in chosen)
        allocation[task] = len(chosen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in selected:
        output_row = {"prompt": row["prompt"], "answer": row["answer"]}
        if args.include_task:
            output_row["task"] = row["task"]
        output_rows.append(output_row)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows
    ))
    manifest = {
        "method": "vanilla-replay", "stage": args.stage,
        "buffer_size": len(selected), "allocation": allocation,
        "seed": args.seed, "include_task": args.include_task,
        "task_order": list(TASKS),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
