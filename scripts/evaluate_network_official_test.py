#!/usr/bin/env python3
"""Final evaluation on official UNSW-NB15 test data with a fixed threshold.

The official testing CSV is used only for scoring and metric reporting. Model
fitting uses the complete training CSV, and the threshold was selected earlier
using a validation fold from that training CSV.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import sys

from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketshield.network.anomaly_detector import NetworkAnomalyDetector  # noqa: E402
from pocketshield.network.schema import (  # noqa: E402
    ATTACK_CATEGORY_COLUMN,
    LABEL_COLUMN,
    MODEL_FEATURE_COLUMNS,
    select_model_features,
    validate_csv_schema,
)

TRAINING_CSV = ROOT / "data" / "network" / "raw" / "UNSW_NB15_training-set.csv"
TESTING_CSV = ROOT / "data" / "network" / "raw" / "UNSW_NB15_testing-set.csv"
SELECTED_THRESHOLD = 0.477573100437


@dataclass(frozen=True)
class CategoryResult:
    category: str
    attacks: int
    detected: int
    missed: int
    recall: float


@dataclass
class OfficialTestResult:
    records: int
    normal_records: int
    attack_records: int
    threshold: float
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    roc_auc: float | None
    average_precision: float | None
    category_results: list[CategoryResult]
    detector: NetworkAnomalyDetector


def load_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV after validating its exact header and row shape."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            schema = validate_csv_schema(reader.fieldnames or ())
            if not schema.is_valid:
                raise ValueError(
                    f"CSV schema mismatch in {path}: missing={schema.missing_columns}, "
                    f"unexpected={schema.unexpected_columns}, duplicates={schema.duplicate_columns}"
                )
            rows = []
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV row {row_number} in {path}: field count does not match header")
                rows.append(row)
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV quoting or syntax in {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read CSV {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def evaluate_official_test_rows(
    training_rows: Iterable[Mapping[str, object]],
    testing_rows: Iterable[Mapping[str, object]],
    *,
    detector: NetworkAnomalyDetector | None = None,
) -> OfficialTestResult:
    """Fit on training rows and evaluate testing rows at the fixed threshold."""
    training = list(training_rows)
    testing = list(testing_rows)
    if not training:
        raise ValueError("Training CSV contains no data rows")
    if not testing:
        raise ValueError("Testing CSV contains no data rows")

    # Validate training labels/features and test labels/features before fitting.
    for row_number, row in enumerate(training, start=2):
        _validate_row(row, row_number, "training")
    labels: list[int] = []
    categories: list[str] = []
    test_feature_rows: list[dict[str, object]] = []
    for row_number, row in enumerate(testing, start=2):
        if not isinstance(row, Mapping):
            raise ValueError(f"Testing row {row_number} must be a mapping of column names to values")
        labels.append(_parse_label(row.get(LABEL_COLUMN), row_number, "testing"))
        if ATTACK_CATEGORY_COLUMN not in row:
            raise ValueError(f"Testing row {row_number} is missing {ATTACK_CATEGORY_COLUMN}")
        category = str(row[ATTACK_CATEGORY_COLUMN]).strip()
        categories.append(category or "(missing)")
        try:
            test_feature_rows.append(select_model_features(row))
        except ValueError as exc:
            raise ValueError(f"Testing row {row_number}: {exc}") from exc

    model = detector if detector is not None else NetworkAnomalyDetector()
    model.fit(training)
    scores = model.anomaly_scores(test_feature_rows)
    predictions = [score >= SELECTED_THRESHOLD for score in scores]

    tp = sum(label == 1 and predicted for label, predicted in zip(labels, predictions))
    tn = sum(label == 0 and not predicted for label, predicted in zip(labels, predictions))
    fp = sum(label == 0 and predicted for label, predicted in zip(labels, predictions))
    fn = sum(label == 1 and not predicted for label, predicted in zip(labels, predictions))
    accuracy = _ratio(tp + tn, len(labels))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1_score = _ratio(2 * precision * recall, precision + recall)
    false_positive_rate = _ratio(fp, fp + tn)

    category_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for label, category, predicted in zip(labels, categories, predictions):
        if label == 1:
            category_counts[category][0] += 1
            category_counts[category][1] += int(predicted)
    category_results = [
        CategoryResult(category, count[0], count[1], count[0] - count[1], _ratio(count[1], count[0]))
        for category, count in sorted(category_counts.items())
    ]

    roc_auc = average_precision = None
    if len(set(labels)) == 2:
        roc_auc = float(roc_auc_score(labels, scores))
        average_precision = float(average_precision_score(labels, scores))

    return OfficialTestResult(
        records=len(testing),
        normal_records=sum(label == 0 for label in labels),
        attack_records=sum(label == 1 for label in labels),
        threshold=SELECTED_THRESHOLD,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        false_positive_rate=false_positive_rate,
        roc_auc=roc_auc,
        average_precision=average_precision,
        category_results=category_results,
        detector=model,
    )


def _validate_row(row: Mapping[str, object], row_number: int, dataset: str) -> None:
    if not isinstance(row, Mapping):
        raise ValueError(f"{dataset.title()} row {row_number} must be a mapping")
    _parse_label(row.get(LABEL_COLUMN), row_number, dataset)
    try:
        select_model_features(row)
    except ValueError as exc:
        raise ValueError(f"{dataset.title()} row {row_number}: {exc}") from exc


def _parse_label(value: object, row_number: int, dataset: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{dataset.title()} row {row_number} has invalid binary label {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{dataset.title()} row {row_number} has invalid binary label {value!r}") from exc
    if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(f"{dataset.title()} row {row_number} has invalid binary label {value!r}")
    return int(numeric)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _print_report(result: OfficialTestResult) -> None:
    print("PocketShield official UNSW-NB15 testing evaluation")
    print(f"Threshold: {result.threshold:.12g} (score >= threshold predicts Attack)")
    print("Threshold source: selected using the training CSV validation fold; not official test data.")
    print("The official testing data was used only for scoring and final metric reporting.")
    print(f"Testing records: {result.records:,}")
    print(f"  Normal records: {result.normal_records:,}")
    print(f"  Attack records: {result.attack_records:,}")
    print("Confusion counts:")
    print(f"  TP: {result.tp:,}  TN: {result.tn:,}  FP: {result.fp:,}  FN: {result.fn:,}")
    print("Metrics:")
    print(f"  Accuracy: {result.accuracy:.6f}")
    print(f"  Precision: {result.precision:.6f}")
    print(f"  Recall: {result.recall:.6f}")
    print(f"  F1-score: {result.f1_score:.6f}")
    print(f"  False Positive Rate: {result.false_positive_rate:.6f}")
    print(f"  ROC-AUC: {result.roc_auc:.6f}" if result.roc_auc is not None else "  ROC-AUC: unavailable (one test class)")
    print(f"  Average Precision: {result.average_precision:.6f}" if result.average_precision is not None else "  Average Precision: unavailable (one test class)")
    print("Attack-category detection (attack_cat used only for reporting):")
    print("  attack_cat                 Attacks  Detected  Missed  Recall")
    for category in result.category_results:
        print(f"  {category.category:<25} {category.attacks:8,} {category.detected:9,} {category.missed:7,} {category.recall:7.4f}")


def main() -> int:
    try:
        training_rows = load_csv(TRAINING_CSV)
        testing_rows = load_csv(TESTING_CSV)
        result = evaluate_official_test_rows(training_rows, testing_rows)
    except (OSError, csv.Error, ValueError) as exc:
        print(f"Official test evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Training records loaded: {len(training_rows):,}")
    print(f"Training normal records used by detector: {result.detector.n_normal_training_samples_:,}")
    _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
