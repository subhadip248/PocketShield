#!/usr/bin/env python3
"""Evaluate PocketShield's offline SMS detector on a labeled TSV dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "SMSSpamCollection"

# Allow running this script directly from a checkout without installation.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pocketshield.sms.sms_detector import analyze_sms  # noqa: E402


SPAM_THRESHOLD = 30


@dataclass(frozen=True)
class Evaluation:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    skipped_rows: int = 0
    skipped_messages: int = 0

    @property
    def evaluated(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> float:
        return _divide(self.tp + self.tn, self.evaluated)

    @property
    def precision(self) -> float:
        return _divide(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _divide(self.tp, self.tp + self.fn)

    @property
    def f1_score(self) -> float:
        return _divide(2 * self.precision * self.recall, self.precision + self.recall)

    @property
    def false_positive_rate(self) -> float:
        return _divide(self.fp, self.fp + self.tn)


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_rows(rows: Iterable[tuple[str, str]]) -> Evaluation:
    """Evaluate valid (label, message) pairs; spam is the positive class."""
    tp = tn = fp = fn = skipped_messages = 0
    for label, message in rows:
        result = analyze_sms(message)
        score = result.get("risk_score")
        if result.get("analysis_status") != "analyzed" or not isinstance(score, (int, float)):
            skipped_messages += 1
            continue

        predicted_spam = score >= SPAM_THRESHOLD
        actual_spam = label == "spam"
        if actual_spam and predicted_spam:
            tp += 1
        elif not actual_spam and not predicted_spam:
            tn += 1
        elif not actual_spam and predicted_spam:
            fp += 1
        else:
            fn += 1
    return Evaluation(tp, tn, fp, fn, skipped_messages=skipped_messages)


def read_dataset(path: Path, warnings: TextIO = sys.stderr) -> Evaluation:
    """Read the UCI label<TAB>message file, warning and skipping bad rows."""
    rows: list[tuple[str, str]] = []
    skipped_rows = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as dataset:
            for line_number, raw_line in enumerate(dataset, start=1):
                line = raw_line.rstrip("\r\n")
                if "\t" not in line:
                    print(f"Warning: skipping malformed row {line_number} (missing tab separator).", file=warnings)
                    skipped_rows += 1
                    continue
                label, message = line.split("\t", 1)
                label = label.strip().lower()
                if label not in {"ham", "spam"}:
                    print(f"Warning: skipping malformed row {line_number} (label must be ham or spam).", file=warnings)
                    skipped_rows += 1
                    continue
                rows.append((label, message))
    except OSError as exc:
        raise OSError(f"cannot read dataset: {exc}") from exc

    result = evaluate_rows(rows)
    return Evaluation(
        result.tp, result.tn, result.fp, result.fn,
        skipped_rows=skipped_rows,
        skipped_messages=result.skipped_messages,
    )


def print_report(result: Evaluation, output: TextIO = sys.stdout) -> None:
    """Print confusion counts and rates in a compact, readable report."""
    print("PocketShield SMS detector evaluation", file=output)
    print(f"Threshold: spam when risk_score >= {SPAM_THRESHOLD}", file=output)
    print(f"Evaluated messages: {result.evaluated}", file=output)
    print("\nConfusion matrix (spam is positive):", file=output)
    print(f"  TP: {result.tp}    FP: {result.fp}", file=output)
    print(f"  FN: {result.fn}    TN: {result.tn}", file=output)
    print("\nMetrics:", file=output)
    print(f"  Accuracy:           {result.accuracy:.4f}", file=output)
    print(f"  Precision:          {result.precision:.4f}", file=output)
    print(f"  Recall:             {result.recall:.4f}", file=output)
    print(f"  F1-score:           {result.f1_score:.4f}", file=output)
    print(f"  False-positive rate:{result.false_positive_rate:>9.4f}", file=output)
    if result.skipped_rows or result.skipped_messages:
        print("\nSkipped:", file=output)
        print(f"  Malformed rows:     {result.skipped_rows}", file=output)
        print(f"  Invalid messages:   {result.skipped_messages}", file=output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", nargs="?", type=Path, default=DEFAULT_DATASET,
        help=f"tab-separated label/message file (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args(argv)

    if not args.dataset.is_file():
        print(
            f"Dataset not found: {args.dataset}\n"
            "Place the UCI SMS Spam Collection file at data/raw/SMSSpamCollection "
            "or pass its path as an argument.",
            file=sys.stderr,
        )
        return 1
    try:
        result = read_dataset(args.dataset)
    except OSError as exc:
        print(f"Unable to read dataset {args.dataset}: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset: {args.dataset}")
    print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
