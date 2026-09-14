#!/usr/bin/env python3
"""Train a ProMoT prompt, model, or joint phase on one TRACE task.

The standalone ProMoT baseline uses prompt-first/model-second training. LAPS
uses the same primitive and consumes both the learned prompt tensor and the
full-fine-tuned backbone when assembling its incremental Bezier simplex.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from peft import PromptTuningConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from trace_task_protocol import TASK_PROMPTS, ensure_task_prompt
from train_trace_pytorch_sft_replay_stage import (
    CompletionCollator,
    CompletionDataset,
    assign_replay_tasks,
    read_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("prompt", "model", "joint"), required=True,
    )
    parser.add_argument("--task", choices=TASK_PROMPTS, required=True)
    parser.add_argument("--stage", type=int, required=True, choices=range(8))
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--buffer", type=Path)
    parser.add_argument(
        "--slow-replay-sft", action="store_true",
        help=(
            "For the model phase, train the plain Slow backbone on the exact "
            "concatenation of current-task and gold-replay rows. The combined "
            "dataset is shuffled by Trainer exactly as in the PyTorch "
            "SFT+Replay baseline; the task prompt is recalibrated afterwards."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument("--model-learning-rate", type=float, default=1e-5)
    parser.add_argument("--prompt-learning-rate", type=float, default=0.3)
    parser.add_argument("--joint-prompt-learning-rate", type=float, default=0.03)
    parser.add_argument("--prompt-length", type=int, default=30)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prompt_embedding(peft_model):
    adapter = peft_model.active_adapter
    if isinstance(adapter, (list, tuple)):
        adapter = adapter[0]
    return peft_model.prompt_encoder[adapter].embedding.weight


def prompt_payload(model, args):
    return {
        "prompt_embeddings": prompt_embedding(model).detach().float().cpu(),
        "prompt_length": args.prompt_length,
        "task": args.task,
        "stage": args.stage,
        "placement": "before_input",
        "seed": args.seed,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.prompt_file.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    rows = read_rows(args.data_root / args.task / "train.json")
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.max_train_samples]
    current_rows = [{
        "prompt": ensure_task_prompt(args.task, str(row["prompt"])),
        "answer": str(row["answer"]),
    } for row in rows]
    replay_rows = []
    if args.slow_replay_sft:
        if args.phase != "model":
            raise ValueError("--slow-replay-sft is valid only for --phase model")
        if args.stage > 0 and args.buffer is None:
            raise ValueError("stages after C-STANCE require --buffer with --slow-replay-sft")
        assigned = assign_replay_tasks(
            read_rows(args.buffer), args.buffer, args.task,
        )
        replay_rows = [{
            "prompt": ensure_task_prompt(
                str(row.get("task") or row.get("_source_task") or args.task),
                str(row["prompt"]),
            ),
            "answer": str(row["answer"]),
        } for row in assigned]
    train_rows = current_rows + replay_rows
    dataset = CompletionDataset(train_rows, tokenizer, args.max_length)

    backbone = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    )
    backbone.config.use_cache = False
    plain_replay_slow = args.phase == "model" and args.slow_replay_sft
    model = get_peft_model(backbone, PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.prompt_length,
        prompt_tuning_init="RANDOM",
    ))

    if args.phase in {"model", "joint"}:
        payload = torch.load(
            args.prompt_file, map_location="cpu", weights_only=True,
        )
        stored = payload["prompt_embeddings"]
        if tuple(stored.shape) != tuple(prompt_embedding(model).shape):
            raise ValueError(
                f"prompt shape mismatch: stored={tuple(stored.shape)}, "
                f"model={tuple(prompt_embedding(model).shape)}"
            )
        with torch.no_grad():
            prompt_embedding(model).copy_(stored)
        if plain_replay_slow:
            # Match the published PyTorch SFT+Replay baseline exactly in the
            # Slow graph: no virtual tokens are inserted for either current
            # or replay rows.  The previously learned task prompt remains in
            # ``prompt_file`` and is recalibrated against this Slow afterwards.
            model = model.get_base_model()
            model.requires_grad_(True)
        else:
            model.get_base_model().requires_grad_(True)
            model.prompt_encoder.requires_grad_(args.phase == "joint")
    else:
        model.get_base_model().requires_grad_(False)
        model.prompt_encoder.requires_grad_(True)

    learning_rate = (
        args.prompt_learning_rate if args.phase == "prompt"
        else args.model_learning_rate
    )
    optimizer = None
    if args.phase == "joint":
        prompt_params = [
            parameter for parameter in model.prompt_encoder.parameters()
            if parameter.requires_grad
        ]
        prompt_ids = {id(parameter) for parameter in prompt_params}
        slow_params = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in prompt_ids
        ]
        optimizer = torch.optim.AdamW([
            {"params": slow_params, "lr": args.model_learning_rate},
            {"params": prompt_params, "lr": args.joint_prompt_learning_rate},
        ], weight_decay=0.0)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / f".{args.phase}_trainer"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=learning_rate,
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
            max_steps=args.max_steps,
        ),
        train_dataset=dataset,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
        optimizers=(optimizer, None),
    )
    result = trainer.train()
    trainer.accelerator.wait_for_everyone()
    unwrapped = trainer.accelerator.unwrap_model(trainer.model_wrapped)

    if trainer.is_world_process_zero() and not args.skip_save:
        if args.phase in {"prompt", "joint"}:
            torch.save(prompt_payload(unwrapped, args), args.prompt_file)
            if args.phase == "prompt":
                (args.output_dir / "PROMPT_COMPLETE").touch()
        if args.phase in {"model", "joint"}:
            base = unwrapped if plain_replay_slow else unwrapped.get_base_model()
            base.config.use_cache = True
            base.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
            saved_prompt = (
                payload if plain_replay_slow else prompt_payload(unwrapped, args)
            )
            torch.save(saved_prompt, args.output_dir / "promot_soft_prompt.pt")
            (args.output_dir / "promot_config.json").write_text(json.dumps({
                "method": "ProMoT",
                "stage": args.stage,
                "task": args.task,
                "prompt_length": args.prompt_length,
                "prompt_learning_rate": args.prompt_learning_rate,
                "model_learning_rate": args.model_learning_rate,
                "joint_prompt_learning_rate": (
                    args.joint_prompt_learning_rate
                    if args.phase == "joint" else None
                ),
                "prompt_placement": "before_input",
                "epochs_per_phase": args.epochs,
                "max_train_samples": args.max_train_samples,
                "tokenized_rows": len(dataset),
                "max_length": args.max_length,
                "per_device_batch_size": args.per_device_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "lr_scheduler": "linear",
                "warmup_ratio": 0.0,
                "weight_decay": 0.0,
                "precision": "bf16",
                "seed": args.seed,
                "replay_rows": len(replay_rows),
                "slow_replay_sft": args.slow_replay_sft,
                "slow_training_graph": (
                    "plain_current_plus_replay_concatenation"
                    if plain_replay_slow else "promot_current_task_prompt"
                ),
            }, indent=2) + "\n")
            # LAPS and standalone ProMoT use different resume markers.
            (args.output_dir / "MODEL_COMPLETE").touch()
            (args.output_dir / "STAGE_COMPLETE").touch()

        metrics = dict(result.metrics)
        metrics.update({
            "method": "ProMoT primitive for LAPS",
            "phase": args.phase,
            "task": args.task,
            "stage": args.stage,
            "current_rows": len(current_rows),
            "tokenized_rows": len(dataset),
            "rows": len(dataset),
            "replay_rows": len(replay_rows),
            "slow_replay_sft": args.slow_replay_sft,
            "slow_training_graph": (
                "plain_current_plus_replay_concatenation"
                if plain_replay_slow else "promot_current_task_prompt"
            ),
            "prompt_length": args.prompt_length,
            "learning_rate": learning_rate,
            "joint_prompt_learning_rate": (
                args.joint_prompt_learning_rate
                if args.phase == "joint" else None
            ),
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
                if torch.cuda.is_available() else 0.0
            ),
        })
        public_name = (
            "prompt_phase_metrics.json"
            if args.phase == "prompt" else "stage_metrics.json"
        )
        laps_name = f"{args.phase}_metrics.json"
        serialized = json.dumps(metrics, indent=2, default=float) + "\n"
        (args.output_dir / public_name).write_text(serialized)
        (args.output_dir / laps_name).write_text(serialized)


if __name__ == "__main__":
    main()
