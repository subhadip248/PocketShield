"""Group-aware train/validation evaluation for UNSW-NB15 training data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import math

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from pocketshield.network.anomaly_detector import NetworkAnomalyDetector
from pocketshield.network.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL_COLUMN,
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    select_model_features,
)


_NUMERIC_FEATURES = frozenset(NUMERICAL_FEATURES + BINARY_FEATURES)


@dataclass(frozen=True)
class SplitSummary:
    """Record, class, group, and conflicting-group counts for one split."""

    records: int
    groups: int
    normal_records: int
    attack_records: int
    conflicting_label_groups: int


@dataclass
class ThresholdMetrics:
    """Confusion counts and threshold-dependent validation metrics."""

    threshold: float
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float


@dataclass
class NetworkEvaluationResult:
    """Results of one group-aware training/validation evaluation."""

    training_summary: SplitSummary
    validation_summary: SplitSummary
    roc_auc: float | None
    average_precision: float | None
    validation_scores: list[float]
    detector: NetworkAnomalyDetector
    validation_labels: list[int] = field(default_factory=list)
    threshold_candidates: list[ThresholdMetrics] = field(default_factory=list)
    selected_threshold: ThresholdMetrics | None = None


def evaluate_thresholds(
    labels: Iterable[object], scores: Iterable[float], thresholds: Iterable[float]
) -> list[ThresholdMetrics]:
    """Evaluate candidate thresholds; score >= threshold predicts Attack."""
    materialized_labels = [_parse_label(label) for label in labels]
    materialized_scores = [_finite_float(score, "score") for score in scores]
    materialized_thresholds = sorted(
        {_finite_float(threshold, "threshold") for threshold in thresholds}
    )
    if not materialized_labels or not materialized_thresholds:
        raise ValueError("labels/scores and thresholds must be non-empty")
    if len(materialized_labels) != len(materialized_scores):
        raise ValueError("labels and scores must have the same length")
    results = []
    for threshold in materialized_thresholds:
        tp = tn = fp = fn = 0
        for label, score in zip(materialized_labels, materialized_scores):
            predicted = score >= threshold
            if label == 1 and predicted:
                tp += 1
            elif label == 0 and not predicted:
                tn += 1
            elif label == 0:
                fp += 1
            else:
                fn += 1
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        results.append(ThresholdMetrics(
            threshold=threshold, tp=tp, tn=tn, fp=fp, fn=fn,
            precision=precision, recall=recall,
            f1_score=_ratio(2 * precision * recall, precision + recall),
            false_positive_rate=_ratio(fp, fp + tn),
        ))
    return results


def select_threshold(metrics: Iterable[ThresholdMetrics]) -> ThresholdMetrics:
    """Choose by F1, Precision, lower FPR, then higher threshold."""
    candidates = list(metrics)
    if not candidates:
        raise ValueError("At least one threshold result is required")
    return max(candidates, key=lambda item: (
        item.f1_score, item.precision, -item.false_positive_rate, item.threshold
    ))


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {name} {value!r}; expected a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {name} {value!r}; expected a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"Invalid {name} {value!r}; expected a finite number")
    return number


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def canonical_feature_key(row: Mapping[str, object]) -> tuple[object, ...]:
    """Return a stable key for all 42 features, normalizing numeric spelling.

    For example, numeric values ``"1"``, ``"1.0"`` and ``"1e0"`` produce
    the same key. Categorical values are trimmed but otherwise kept as text;
    the service placeholder ``-`` remains a valid, distinct category.
    """
    features = select_model_features(row)
    key: list[object] = []
    for name in MODEL_FEATURE_COLUMNS:
        value = features[name]
        if name in _NUMERIC_FEATURES:
            key.append(("numeric", _canonical_number(value)))
        elif name in CATEGORICAL_FEATURES:
            key.append(("category", "" if value is None else str(value).strip()))
        else:  # Defensive: schema feature groups should cover every feature.
            key.append(("text", "" if value is None else str(value).strip()))
    return tuple(key)


def _canonical_number(value: object) -> object:
    if value is None or (isinstance(value, str) and not value.strip()):
        return ("missing",)
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return ("invalid", str(value).strip())
    if not number.is_finite():
        return ("nonfinite", str(number).lower())
    if number == 0:
        return Decimal(0)
    return number.normalize()


def split_training_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    """Create one deterministic, approximately 80/20 grouped stratified split.

    Duplicate feature vectors always receive the same group ID, even when
    their labels conflict. The splitter stratifies record labels while
    respecting those groups.
    """
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot split an empty training dataset")
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise ValueError("random_state must be an integer")

    labels: list[int] = []
    group_index: dict[tuple[object, ...], int] = {}
    groups: list[int] = []
    for row in materialized:
        if LABEL_COLUMN not in row:
            raise ValueError(f"Training row is missing required column: {LABEL_COLUMN}")
        label = _parse_label(row[LABEL_COLUMN])
        key = canonical_feature_key(row)
        if key not in group_index:
            group_index[key] = len(group_index)
        labels.append(label)
        groups.append(group_index[key])

    if len(set(labels)) < 2:
        raise ValueError("Stratified splitting requires both normal and attack records")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    split_candidates = list(splitter.split([0] * len(materialized), labels, groups))
    if not split_candidates:
        raise ValueError("Unable to create a group-aware train/validation split")
    all_classes = set(labels)
    best_candidate = None
    best_quality = None
    for fold_number, (training_indices, validation_indices) in enumerate(split_candidates):
        training_classes = {labels[int(index)] for index in training_indices}
        validation_classes = {labels[int(index)] for index in validation_indices}
        validation_fraction_error = abs(
            len(validation_indices) / len(materialized) - 1 / n_splits
        )
        quality = (
            validation_classes == all_classes,
            0 in training_classes,
            -validation_fraction_error,
            training_classes == all_classes,
            -fold_number,
        )
        if best_quality is None or quality > best_quality:
            best_candidate = (training_indices, validation_indices)
            best_quality = quality
    assert best_candidate is not None
    training_indices, validation_indices = best_candidate
    return (
        [materialized[int(index)] for index in training_indices],
        [materialized[int(index)] for index in validation_indices],
    )


def summarize_split(rows: Iterable[Mapping[str, object]]) -> SplitSummary:
    """Count records, unique feature groups, class totals, and label conflicts."""
    materialized = list(rows)
    labels_by_group: dict[tuple[object, ...], set[int]] = defaultdict(set)
    normal_records = 0
    attack_records = 0
    for row in materialized:
        if LABEL_COLUMN not in row:
            raise ValueError(f"Training row is missing required column: {LABEL_COLUMN}")
        label = _parse_label(row[LABEL_COLUMN])
        labels_by_group[canonical_feature_key(row)].add(label)
        if label == 0:
            normal_records += 1
        else:
            attack_records += 1
    return SplitSummary(
        records=len(materialized),
        groups=len(labels_by_group),
        normal_records=normal_records,
        attack_records=attack_records,
        conflicting_label_groups=sum(len(labels) > 1 for labels in labels_by_group.values()),
    )


def evaluate_training_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    detector: NetworkAnomalyDetector | None = None,
) -> NetworkEvaluationResult:
    """Fit on a training-row split and evaluate validation scores and thresholds.

    Both labels are passed to ``detector.fit``; its existing implementation
    retains only normal rows internally. The validation set is scored with
    both classes. Threshold candidates are derived only from validation scores,
    evaluated against validation labels, and one is selected by F1, Precision,
    false-positive rate, then threshold.
    """
    materialized = list(rows)
    training_rows, validation_rows = split_training_rows(
        materialized,
        n_splits=n_splits,
        random_state=random_state,
    )
    model = detector if detector is not None else NetworkAnomalyDetector(random_state=random_state)
    model.fit(training_rows)
    scores = model.anomaly_scores(validation_rows)
    labels = [_parse_label(row[LABEL_COLUMN]) for row in validation_rows]
    thresholds = sorted(set(scores))
    thresholds.append(math.nextafter(max(scores), math.inf))
    threshold_candidates = evaluate_thresholds(labels, scores, thresholds)
    selected_threshold = select_threshold(threshold_candidates)

    roc_auc: float | None = None
    average_precision: float | None = None
    if len(set(labels)) == 2 and all(math.isfinite(score) for score in scores):
        roc_auc = float(roc_auc_score(labels, scores))
        average_precision = float(average_precision_score(labels, scores))

    return NetworkEvaluationResult(
        training_summary=summarize_split(training_rows),
        validation_summary=summarize_split(validation_rows),
        roc_auc=roc_auc,
        average_precision=average_precision,
        validation_scores=scores,
        detector=model,
        validation_labels=labels,
        threshold_candidates=threshold_candidates,
        selected_threshold=selected_threshold,
    )


def _parse_label(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid label {value!r}; expected 0 (normal) or 1 (attack)")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid label {value!r}; expected 0 (normal) or 1 (attack)"
        ) from exc
    if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(f"Invalid label {value!r}; expected 0 (normal) or 1 (attack)")
    return int(numeric)

