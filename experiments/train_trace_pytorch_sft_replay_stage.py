#!/usr/bin/env python3
"""Full-parameter TRACE SFT with optional fixed-budget gold replay.

This backend uses only PyTorch, Transformers Trainer, and Accelerate.  It is
the lightweight alternative to the ms-swift/DeepSpeed baseline and keeps the
same completion-only SFT objective and effective global batch.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from trace_task_protocol import TASK_PROMPTS, ensure_task_prompt


class CompletionDataset(Dataset):
    """Tokenize chat prompts and supervise response tokens only."""

    def __init__(self, rows, tokenizer, max_length: int):
        self.items = []
        for row in rows:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer(
                rendered, add_special_tokens=False,
            ).input_ids
            answer_ids = tokenizer(
                str(row["answer"]), add_special_tokens=False,
            ).input_ids + [tokenizer.eos_token_id]
            input_ids = (prompt_ids + answer_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
            if any(label != -100 for label in labels):
                self.items.append((input_ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        width = max(len(item[0]) for item in batch)
        input_ids, labels, attention_masks = [], [], []
        for ids, target in batch:
            padding = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * padding)
            labels.append(target + [-100] * padding)
            attention_masks.append([1] * len(ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_masks),
        }


def read_rows(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    if path.suffix == ".jsonl":
        return [
            json.loads(line) for line in path.read_text().splitlines()
            if line.strip()
        ]
    return json.loads(path.read_text())


def assign_replay_tasks(
    rows: list[dict], buffer_path: Path | None, fallback_task: str,
) -> list[dict]:
    """Restore task identities for published buffers with task-less rows."""
    if not rows or any(row.get("task") for row in rows):
        return rows
    manifest_path = (
        buffer_path.with_name("buffer.manifest.json")
        if buffer_path is not None else None
    )
    if manifest_path is not None and manifest_path.exists():
        allocation = json.loads(manifest_path.read_text()).get("allocation", {})
        if sum(int(count) for count in allocation.values()) != len(rows):
            raise ValueError(
                f"Buffer allocation does not match {len(rows)} rows: "
                f"{manifest_path}"
            )
        offset = 0
        for task, count in allocation.items():
            for row in rows[offset:offset + int(count)]:
                row["_source_task"] = task
            offset += int(count)
        return rows
    for row in rows:
        prompt = str(row["prompt"])
        matches = [
            task for task, prefix in TASK_PROMPTS.items()
            if prefix and prompt.startswith(prefix)
        ]
        row["_source_task"] = matches[0] if len(matches) == 1 else fallback_task
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASK_PROMPTS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--buffer", type=Path)
    parser.add_argument("--buffer-task", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    current = read_rows(args.data_root / args.task / "train.json")
    random.Random(args.seed).shuffle(current)
    current = current[:args.max_train_samples]
    current = [
        {
            "prompt": ensure_task_prompt(args.task, str(row["prompt"])),
            "answer": str(row["answer"]),
        }
        for row in current
    ]
    replay = assign_replay_tasks(
        read_rows(args.buffer), args.buffer, args.buffer_task or args.task,
    )
    replay = [
        {
            "prompt": ensure_task_prompt(
                str(
                    row.get("task") or row.get("_source_task")
                    or args.buffer_task or args.task
                ),
                str(row["prompt"]),
            ),
            "answer": str(row["answer"]),
        }
        for row in replay
    ]
    dataset = CompletionDataset(current + replay, tokenizer, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        weight_decay=0.0,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
    )
    result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    if trainer.is_world_process_zero():
        metrics = {
            **result.metrics,
            "task": args.task,
            "current_rows": len(current),
            "replay_rows": len(replay),
            "tokenized_rows": len(dataset),
        }
        (args.output_dir / "stage_metrics.json").write_text(
            json.dumps(metrics, indent=2, default=float) + "\n"
        )
        (args.output_dir / "STAGE_COMPLETE").touch()


if __name__ == "__main__":
    main()
