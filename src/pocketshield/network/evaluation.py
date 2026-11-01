"""Group-aware train/validation evaluation for UNSW-NB15 training data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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
class NetworkEvaluationResult:
    """Results of one group-aware training/validation evaluation."""

    training_summary: SplitSummary
    validation_summary: SplitSummary
    roc_auc: float | None
    average_precision: float | None
    validation_scores: list[float]
    detector: NetworkAnomalyDetector


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
    """Fit on a split of training CSV rows and evaluate threshold-free metrics.

    Both labels are passed to ``detector.fit``; its existing implementation
    retains only normal rows internally. The validation set is scored with
    both classes. No decision threshold or predictions are produced.
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

