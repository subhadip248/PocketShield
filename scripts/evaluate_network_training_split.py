#!/usr/bin/env python3
"""Evaluate a grouped validation fold created only from the training CSV."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketshield.network.evaluation import evaluate_training_rows  # noqa: E402
from pocketshield.network.schema import validate_csv_schema  # noqa: E402


TRAINING_CSV = ROOT / "data" / "network" / "raw" / "UNSW_NB15_training-set.csv"


def _print_summary(title: str, summary) -> None:
    print(title)
    print(f"  Records: {summary.records:,}")
    print(f"  Feature groups: {summary.groups:,}")
    print(f"  Normal records: {summary.normal_records:,}")
    print(f"  Attack records: {summary.attack_records:,}")
    print(f"  Conflicting-label groups: {summary.conflicting_label_groups:,}")


def main() -> int:
    if not TRAINING_CSV.is_file():
        print(f"Training dataset not found: {TRAINING_CSV}", file=sys.stderr)
        print("Place UNSW_NB15_training-set.csv in data/network/raw/.", file=sys.stderr)
        return 1

    try:
        with TRAINING_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            schema_result = validate_csv_schema(reader.fieldnames or ())
            if not schema_result.is_valid:
                raise ValueError(
                    "Training CSV schema mismatch: "
                    f"missing={schema_result.missing_columns}, "
                    f"unexpected={schema_result.unexpected_columns}, "
                    f"duplicates={schema_result.duplicate_columns}"
                )
            rows = list(reader)

        result = evaluate_training_rows(rows, n_splits=5, random_state=42)
    except (OSError, csv.Error, ValueError) as exc:
        print(f"Could not evaluate training dataset: {exc}", file=sys.stderr)
        return 1

    print("PocketShield network training/validation evaluation")
    print("Source: UNSW_NB15_training-set.csv only")
    print("Split: StratifiedGroupKFold, 5 folds, shuffled, random_state=42")
    _print_summary("Training fold:", result.training_summary)
    _print_summary("Validation fold:", result.validation_summary)
    print("Threshold-independent validation metrics:")
    print(f"  ROC-AUC: {result.roc_auc:.6f}" if result.roc_auc is not None else "  ROC-AUC: unavailable (one validation class)")
    print(
        f"  Average Precision: {result.average_precision:.6f}"
        if result.average_precision is not None
        else "  Average Precision: unavailable (one validation class)"
    )
    print(
        "Validation threshold candidates (score >= threshold predicts Attack; "
        f"showing top 10 of {len(result.threshold_candidates)} by selection criterion):"
    )
    print("  threshold       TP       TN       FP       FN  Precision  Recall  F1-score      FPR")
    ranked_candidates = sorted(
        result.threshold_candidates,
        key=lambda item: (
            item.f1_score, item.precision, -item.false_positive_rate, item.threshold
        ),
        reverse=True,
    )
    displayed = ranked_candidates[:10]
    selected = result.selected_threshold
    if selected is not None and selected not in displayed:
        displayed.append(selected)
    for candidate in displayed:
        print(
            f"  {candidate.threshold:9.6f} {candidate.tp:8,} {candidate.tn:8,}"
            f" {candidate.fp:8,} {candidate.fn:8,} {candidate.precision:10.4f}"
            f" {candidate.recall:7.4f} {candidate.f1_score:9.4f}"
            f" {candidate.false_positive_rate:9.4f}"
        )
    print("Selected validation threshold:")
    if selected is None:
        print("  unavailable")
    else:
        print(f"  {selected.threshold:.12g}")
        print("  Criterion: highest F1, then higher Precision, lower FPR, then higher threshold")
        print("  Selection metrics are measured on the validation fold used for tuning.")
    print("  The official testing CSV was not used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

