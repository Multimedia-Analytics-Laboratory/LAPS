#!/usr/bin/env python3
"""TRACE task metrics matching the paper's task-specific metric families."""

from __future__ import annotations

import difflib
import re
from collections import Counter

import sacrebleu
from packaging import version
from rouge_score import rouge_scorer

def first_choice(text: str) -> str:
    marked = re.findall(
        r"(?is)(?:final\s+answer|answer|最终答案|答案)\s*(?:is|是)?\s*[:：]?\s*\**\s*([A-D])(?=\s|[.):,，、*]|$)",
        text,
    )
    if marked:
        return marked[-1].upper()
    standalone = re.findall(
        r"(?im)^\s*(?:[-*#>]\s*)*\**\s*([A-D])(?=\s|[.):,，、*]|$)", text,
    )
    return standalone[-1].upper() if standalone else ""


def numeric_answer(text: str) -> str:
    candidates = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return candidates[-1].replace(",", "") if candidates else ""


def _clean_code(code: str) -> str:
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    for kind, value in re.findall(pattern, code):
        code = code.replace(f"<{kind}_LIT:{value}>", value)
    return code


def _fuzz_ratio(left: str, right: str) -> int:
    # fuzzywuzzy.fuzz.ratio without python-Levenshtein uses this implementation.
    return int(round(100.0 * difflib.SequenceMatcher(None, left, right).ratio()))


def _sari_ngram(source, candidate, references, num_references):
    references_all = [item for reference in references for item in reference]
    reference_counter = Counter(references_all)
    source_counter = Counter(source)
    source_repeated = Counter({item: count * num_references for item, count in source_counter.items()})
    candidate_counter = Counter(candidate)
    candidate_repeated = Counter({item: count * num_references for item, count in candidate_counter.items()})

    keep = source_repeated & candidate_repeated
    keep_good = keep & reference_counter
    keep_all = source_repeated & reference_counter
    keep_precision = 1.0 if not keep else sum(keep_good[x] / keep[x] for x in keep_good) / len(keep)
    keep_recall = 1.0 if not keep_all else sum(keep_good.values()) / sum(keep_all.values())
    keep_score = 0.0 if keep_precision == keep_recall == 0 else 2 * keep_precision * keep_recall / (keep_precision + keep_recall)

    deleted = source_repeated - candidate_repeated
    deleted_good = deleted - reference_counter
    delete_precision = 1.0 if not deleted else sum(deleted_good[x] / deleted[x] for x in deleted_good) / len(deleted)

    added = set(candidate_counter) - set(source_counter)
    added_good = added & set(reference_counter)
    added_all = set(reference_counter) - set(source_counter)
    add_precision = 1.0 if not added else len(added_good) / len(added)
    add_recall = 1.0 if not added_all else len(added_good) / len(added_all)
    add_score = 0.0 if add_precision == add_recall == 0 else 2 * add_precision * add_recall / (add_precision + add_recall)
    return keep_score, delete_precision, add_score


def _normalize(text: str) -> str:
    text = text.lower()
    if version.parse(sacrebleu.__version__).major >= 2:
        return sacrebleu.metrics.bleu._get_tokenizer("13a")()(text)
    return sacrebleu.TOKENIZERS["13a"]()(text)


def _ngrams(tokens: list[str], size: int) -> list[str]:
    return [" ".join(tokens[i:i + size]) for i in range(max(0, len(tokens) - size + 1))]


def _sari_sentence(source: str, candidate: str, references: list[str]) -> float:
    source_tokens = _normalize(source).split(" ")
    candidate_tokens = _normalize(candidate).split(" ")
    reference_tokens = [_normalize(item).split(" ") for item in references]
    scores = []
    for size in range(1, 5):
        scores.append(_sari_ngram(
            _ngrams(source_tokens, size), _ngrams(candidate_tokens, size),
            [_ngrams(item, size) for item in reference_tokens], len(references),
        ))
    keep = sum(item[0] for item in scores) / 4
    delete = sum(item[1] for item in scores) / 4
    add = sum(item[2] for item in scores) / 4
    return 100.0 * (keep + delete + add) / 3


def _simplification_source(prompt: str) -> str:
    if "Paragraph:\n" in prompt:
        return prompt.split("Paragraph:\n", 1)[1].split("\n\nSimplification:", 1)[0]
    return prompt


def paper_score(task: str, predictions: list[str], rows: list[dict]) -> float:
    targets = [str(row["answer"]).strip() for row in rows]
    if task in {"C-STANCE", "FOMC", "ScienceQA"}:
        return 100.0 * sum(first_choice(p) == first_choice(t) for p, t in zip(predictions, targets)) / len(rows)
    if task in {"NumGLUE-cm", "NumGLUE-ds"}:
        return 100.0 * sum(numeric_answer(p) == numeric_answer(t) for p, t in zip(predictions, targets)) / len(rows)
    if task == "MeetingBank":
        scorer = rouge_scorer.RougeScorer(["rougeL"])
        return sum(100.0 * scorer.score(t, p)["rougeL"].fmeasure for p, t in zip(predictions, targets)) / len(rows)
    if task == "Py150":
        return sum(_fuzz_ratio(_clean_code(p), _clean_code(t)) for p, t in zip(predictions, targets)) / len(rows)
    if task == "20Minuten":
        return sum(
            _sari_sentence(_simplification_source(str(row["prompt"])), prediction, [target])
            for row, prediction, target in zip(rows, predictions, targets)
        ) / len(rows)
    raise ValueError(task)
