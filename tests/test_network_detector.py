"""Tests for the normal-only Isolation Forest network detector."""

import math

import pytest

from pocketshield.network.anomaly_detector import NetworkAnomalyDetector
from pocketshield.network.schema import MODEL_FEATURE_COLUMNS


def make_row(**updates):
    row = {column: "0" for column in MODEL_FEATURE_COLUMNS}
    row.update({"proto": "tcp", "service": "-", "state": "FIN", "dbytes": "10"})
    row.update(updates)
    return row


def training_rows():
    # A little normal variation gives the forest a useful reference range.
    rows = [
        make_row(label="0", id=str(index), dbytes=str(10 + index % 5), spkts=str(2 + index % 3))
        for index in range(24)
    ]
    rows.append(
        make_row(
            label="1",
            id="999",
            attack_cat="DoS",
            proto="attack-only-protocol",
            dbytes="999999999",
        )
    )
    return rows


def test_fit_uses_only_normal_records_for_preprocessing_and_forest():
    detector = NetworkAnomalyDetector(n_estimators=40).fit(training_rows())

    assert detector.n_normal_training_samples_ == 24
    assert detector.n_attack_rows_excluded_ == 1
    assert "attack-only-protocol" not in detector.preprocessor.category_mapping_["proto"]
    assert detector.preprocessor.numeric_impute_values_["dbytes"] < 100


def test_detector_is_reproducible_with_the_same_random_state():
    rows = training_rows()[:-1]
    samples = [make_row(dbytes="12"), make_row(dbytes="500000")]

    first = NetworkAnomalyDetector(n_estimators=40, random_state=13).fit(rows)
    second = NetworkAnomalyDetector(n_estimators=40, random_state=13).fit(rows)

    assert first.anomaly_scores(samples) == second.anomaly_scores(samples)


def test_scores_have_expected_shape_are_finite_and_higher_for_an_outlier():
    detector = NetworkAnomalyDetector(n_estimators=100).fit(training_rows()[:-1])
    normal = make_row(dbytes="12", spkts="3")
    outlier = make_row(dbytes="1000000000", spkts="10000000")

    scores = detector.anomaly_scores([normal, outlier])

    assert len(scores) == 2
    assert all(math.isfinite(score) for score in scores)
    assert scores[1] > scores[0]


def test_fit_rejects_empty_input_or_data_without_normal_records():
    detector = NetworkAnomalyDetector()

    with pytest.raises(ValueError, match="empty"):
        detector.fit([])
    with pytest.raises(ValueError, match="no normal"):
        detector.fit([make_row(label="1")])


@pytest.mark.parametrize("label", ["2", "spam", "", "nan", "inf", True])
def test_fit_rejects_invalid_labels(label):
    with pytest.raises(ValueError, match="Invalid label"):
        NetworkAnomalyDetector().fit([make_row(label=label)])


def test_fit_rejects_missing_label_and_missing_feature_columns():
    missing_label = make_row()
    missing_label.pop("label", None)
    with pytest.raises(ValueError, match="label"):
        NetworkAnomalyDetector().fit([missing_label])

    missing_feature = make_row(label="0")
    missing_feature.pop("service")
    with pytest.raises(ValueError, match="service"):
        NetworkAnomalyDetector().fit([missing_feature])


def test_scoring_rejects_pre_fit_empty_and_missing_feature_input():
    detector = NetworkAnomalyDetector()
    with pytest.raises(RuntimeError, match="fit"):
        detector.anomaly_scores([make_row()])

    detector.fit(training_rows()[:-1])
    with pytest.raises(ValueError, match="empty"):
        detector.anomaly_scores([])
    missing_feature = make_row()
    missing_feature.pop("state")
    with pytest.raises(ValueError, match="state"):
        detector.anomaly_scores([missing_feature])

