"""Tests for group-aware UNSW-NB15 training/validation evaluation."""

from pocketshield.network.evaluation import (
    canonical_feature_key,
    evaluate_training_rows,
    split_training_rows,
    summarize_split,
)
from pocketshield.network.schema import MODEL_FEATURE_COLUMNS


def make_row(**updates):
    row = {column: "0" for column in MODEL_FEATURE_COLUMNS}
    row.update({"proto": "tcp", "service": "-", "state": "FIN", "dbytes": "10"})
    row.update(updates)
    return row


def synthetic_training_rows():
    rows = []
    for index in range(30):
        rows.append(
            make_row(
                label="0",
                id=f"normal-{index}",
                attack_cat="Normal",
                dbytes=str(10 + index),
                sbytes=str(100 + index),
            )
        )
    for index in range(30):
        rows.append(
            make_row(
                label="1",
                id=f"attack-{index}",
                attack_cat="DoS",
                dbytes=str(1000 + index),
                sbytes=str(2000 + index),
            )
        )
    # Same complete feature vector as the first normal record, opposite label.
    conflict = dict(rows[0])
    conflict.update({"label": "1", "attack_cat": "Generic", "id": "conflict"})
    rows.append(conflict)
    return rows


def test_canonical_feature_key_normalizes_numbers_and_ignores_metadata():
    first = make_row(dur="1", dbytes="1.0", id="1", label="0", attack_cat="Normal")
    equivalent = make_row(dur="1e0", dbytes="1.000", id="99", label="1", attack_cat="DoS")

    assert canonical_feature_key(first) == canonical_feature_key(equivalent)


def test_split_is_reproducible_approximately_80_20_and_keeps_groups_intact():
    rows = synthetic_training_rows()

    training_a, validation_a = split_training_rows(rows, random_state=19)
    training_b, validation_b = split_training_rows(rows, random_state=19)

    assert training_a == training_b
    assert validation_a == validation_b
    assert 0.15 <= len(validation_a) / len(rows) <= 0.25
    train_keys = {canonical_feature_key(row) for row in training_a}
    validation_keys = {canonical_feature_key(row) for row in validation_a}
    assert train_keys.isdisjoint(validation_keys)
    assert {row["label"] for row in validation_a} == {"0", "1"}


def test_conflicting_label_duplicates_stay_together_and_are_reported():
    rows = synthetic_training_rows()
    training, validation = split_training_rows(rows)
    locations = {}
    for split_name, split_rows in (("train", training), ("validation", validation)):
        for row in split_rows:
            if row["id"] in ("normal-0", "conflict"):
                locations[row["id"]] = split_name

    assert locations["normal-0"] == locations["conflict"]
    assert (
        summarize_split(training).conflicting_label_groups
        + summarize_split(validation).conflicting_label_groups
        == 1
    )


def test_evaluation_fits_complete_fold_and_scores_both_validation_classes():
    result = evaluate_training_rows(synthetic_training_rows(), random_state=7)

    assert result.training_summary.normal_records > 0
    assert result.training_summary.attack_records > 0
    assert result.detector.n_normal_training_samples_ == result.training_summary.normal_records
    assert result.detector.n_attack_rows_excluded_ == result.training_summary.attack_records
    assert len(result.validation_scores) == result.validation_summary.records
    assert result.validation_summary.normal_records > 0
    assert result.validation_summary.attack_records > 0
    assert result.roc_auc is not None
    assert result.average_precision is not None

