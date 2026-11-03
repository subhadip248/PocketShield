"""Synthetic tests for final network test-set evaluation helpers."""

import csv

import pytest

from scripts.evaluate_network_official_test import (
    SELECTED_THRESHOLD,
    evaluate_official_test_rows,
    load_csv,
)
from pocketshield.network.schema import CSV_COLUMNS, MODEL_FEATURE_COLUMNS


def make_row(label, category, **updates):
    row = {column: "0" for column in MODEL_FEATURE_COLUMNS}
    row.update({"proto": "tcp", "service": "-", "state": "FIN"})
    row.update({"label": str(label), "attack_cat": category, "id": "synthetic"})
    row.update(updates)
    return row


class RecordingDetector:
    def __init__(self, scores):
        self.scores = scores
        self.fit_rows = None
        self.scored_rows = None

    def fit(self, rows):
        self.fit_rows = list(rows)
        return self

    def anomaly_scores(self, rows):
        self.scored_rows = list(rows)
        return self.scores


def test_evaluation_uses_fixed_inclusive_threshold_and_reports_attack_categories():
    training = [make_row(0, "Normal", dbytes=str(i + 1)) for i in range(3)]
    testing = [
        make_row(0, "Normal"),
        make_row(1, "DoS"),
        make_row(1, "DoS"),
        make_row(1, "Exploits"),
    ]
    detector = RecordingDetector([SELECTED_THRESHOLD, SELECTED_THRESHOLD - 0.01,
                                  SELECTED_THRESHOLD + 0.01, SELECTED_THRESHOLD + 0.02])

    result = evaluate_official_test_rows(training, testing, detector=detector)

    assert detector.fit_rows == training
    assert len(detector.scored_rows) == len(testing)
    assert all(set(row) == set(MODEL_FEATURE_COLUMNS) for row in detector.scored_rows)
    assert result.threshold == SELECTED_THRESHOLD
    assert (result.records, result.normal_records, result.attack_records) == (4, 1, 3)
    assert (result.tp, result.tn, result.fp, result.fn) == (2, 0, 1, 1)
    assert result.accuracy == 0.5
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.false_positive_rate == 1
    by_category = {item.category: item for item in result.category_results}
    assert (by_category["DoS"].attacks, by_category["DoS"].detected,
            by_category["DoS"].missed, by_category["DoS"].recall) == (2, 1, 1, 0.5)
    assert (by_category["Exploits"].attacks, by_category["Exploits"].detected) == (1, 1)
    assert result.roc_auc is not None
    assert result.average_precision is not None


def test_testing_label_and_required_features_are_validated_before_fit():
    training = [make_row(0, "Normal")]
    detector = RecordingDetector([])
    with pytest.raises(ValueError, match="invalid binary label"):
        evaluate_official_test_rows(training, [make_row("2", "DoS")], detector=detector)
    assert detector.fit_rows is None
    missing_feature = make_row(0, "Normal")
    missing_feature.pop("service")
    with pytest.raises(ValueError, match="service"):
        evaluate_official_test_rows(training, [missing_feature], detector=detector)
    assert detector.fit_rows is None


def test_non_mapping_testing_row_is_rejected_before_fit():
    detector = RecordingDetector([])
    with pytest.raises(ValueError, match="Testing row 2 must be a mapping"):
        evaluate_official_test_rows([make_row(0, "Normal")], [None], detector=detector)
    assert detector.fit_rows is None


def test_ranking_metrics_match_known_values():
    training = [make_row(0, "Normal")]
    testing = [
        make_row(0, "Normal"),
        make_row(0, "Normal"),
        make_row(1, "DoS"),
        make_row(1, "Exploits"),
    ]
    result = evaluate_official_test_rows(
        training,
        testing,
        detector=RecordingDetector([0.1, 0.4, 0.35, 0.8]),
    )
    assert result.roc_auc == pytest.approx(0.75)
    assert result.average_precision == pytest.approx(5 / 6)


def test_ranking_metrics_are_unavailable_for_one_class_testing_labels():
    result = evaluate_official_test_rows(
        [make_row(0, "Normal")],
        [make_row(0, "Normal"), make_row(0, "Normal")],
        detector=RecordingDetector([0.1, 0.4]),
    )
    assert result.roc_auc is None
    assert result.average_precision is None


def test_csv_loader_rejects_invalid_schema_and_malformed_rows(tmp_path):
    schema_path = tmp_path / "bad_schema.csv"
    schema_path.write_text("id,label\n1,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        load_csv(schema_path)

    malformed_path = tmp_path / "malformed.csv"
    with malformed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerow(["only-one-field"])
    with pytest.raises(ValueError, match="Malformed CSV row"):
        load_csv(malformed_path)


def test_csv_loader_rejects_malformed_quoting(tmp_path):
    malformed_path = tmp_path / "malformed_quotes.csv"
    with malformed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        handle.write(",".join(["0"] * (len(CSV_COLUMNS) - 1) + ['"unterminated']) + "\n")
    with pytest.raises(ValueError, match="Malformed CSV quoting or syntax"):
        load_csv(malformed_path)


def test_csv_loader_reports_missing_file_clearly(tmp_path):
    with pytest.raises(ValueError, match="Could not read CSV"):
        load_csv(tmp_path / "missing.csv")
