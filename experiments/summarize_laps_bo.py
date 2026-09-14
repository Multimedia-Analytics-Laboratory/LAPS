#!/usr/bin/env python3
"""Summarize final LAPS BO evaluations and paired continual metrics."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path


_CANONICAL_TASKS = (
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
)
TASKS = tuple(filter(None, os.environ.get(
    "TRACE_TASK_ORDER", ",".join(_CANONICAL_TASKS),
).split(",")))
if len(TASKS) != 8 or set(TASKS) != set(_CANONICAL_TASKS):
    raise ValueError(f"invalid TRACE_TASK_ORDER: {TASKS}")


def summary(values: list[float]) -> dict[str, object]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def is_task_vertex(
    preference: list[float] | None, task_index: int, *, atol: float = 1e-8,
) -> bool:
    """Return whether a selected simplex preference is exactly e_task."""
    if preference is None or len(preference) != len(TASKS):
        return False
    return all(
        abs(float(value) - (1.0 if index == task_index else 0.0)) <= atol
        for index, value in enumerate(preference)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    final_scores: dict[str, list[float]] = {}
    acquisition_scores: dict[str, list[float]] = {}
    selected_preferences: dict[str, list[float] | None] = {}
    final_score_sources: dict[str, str] = {}
    final_vertex_dir = (
        args.run_root / f"stage_{len(TASKS) - 1:02d}_{TASKS[-1]}"
        / "eval_before_fast_temp01_x8"
    )
    has_complete_final_vertex_cache = all(
        (final_vertex_dir / f"{task}_vertex.json").is_file()
        for task in TASKS
    )
    for stage, task in enumerate(TASKS):
        bo_path = args.bo_dir / f"search_{task}.json"
        if not bo_path.is_file():
            raise FileNotFoundError(bo_path)
        bo_payload = json.loads(bo_path.read_text())
        result = bo_payload["tasks"][task]
        selected_preferences[task] = result.get("selected_preference")

        # Fast training freezes all task vertices.  If BO selects the exact
        # task vertex, reuse its already-recorded final-checkpoint evaluation
        # instead of reporting a second stochastic generation for the same
        # model state and preference.
        final_vertex_path = final_vertex_dir / f"{task}_vertex.json"
        if (
            is_task_vertex(selected_preferences[task], stage)
            and has_complete_final_vertex_cache
        ):
            final_vertex = json.loads(final_vertex_path.read_text())
            final_scores[task] = [
                float(value) for value in final_vertex["scores"]
            ]
            final_score_sources[task] = "cached_final_task_vertex"
        else:
            final_scores[task] = [
                float(value) for value in result["test_scores"]
            ]
            final_score_sources[task] = "bo_selected_preference"

        diagonal_path = (
            args.run_root / f"stage_{stage:02d}_{task}"
            / "eval_before_fast_temp01_x8" / f"{task}_vertex.json"
        )
        diagonal = json.loads(diagonal_path.read_text())
        acquisition_scores[task] = [float(value) for value in diagonal["scores"]]

    repeats = {len(values) for values in final_scores.values()}
    repeats.update(len(values) for values in acquisition_scores.values())
    if len(repeats) != 1:
        raise ValueError(f"unaligned repeat counts: {sorted(repeats)}")
    repeat_count = repeats.pop()

    acc = [
        statistics.fmean(final_scores[task][repeat] for task in TASKS)
        for repeat in range(repeat_count)
    ]
    bwt = [
        statistics.fmean(
            final_scores[task][repeat] - acquisition_scores[task][repeat]
            for task in TASKS[:-1]
        )
        for repeat in range(repeat_count)
    ]
    output = {
        "run_root": str(args.run_root),
        "bo_dir": str(args.bo_dir),
        "tasks": {
            task: {
                "selected_preference": selected_preferences[task],
                "final_score_source": final_score_sources[task],
                "at_acquisition": summary(acquisition_scores[task]),
                "at_final": summary(final_scores[task]),
                "backward_change": summary([
                    final - acquired for final, acquired in zip(
                        final_scores[task], acquisition_scores[task], strict=True,
                    )
                ]),
            }
            for task in TASKS
        },
        "metrics": {"ACC": summary(acc), "BWT": summary(bwt)},
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")

    for task in TASKS:
        result = output["tasks"][task]["at_final"]
        print(f"{task}: {result['mean']:.4f} ± {result['std_sample']:.4f}")
    for metric in ("ACC", "BWT"):
        result = output["metrics"][metric]
        print(f"{metric}: {result['mean']:.4f} ± {result['std_sample']:.4f}")


if __name__ == "__main__":
    main()
