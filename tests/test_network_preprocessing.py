"""Tests for UNSW-NB15 preprocessing fitted on training rows only."""

import math

import pytest

from pocketshield.network.preprocessing import (
    LOG1P_FEATURES,
    RARE_CATEGORY,
    UNKNOWN_CATEGORY,
    NetworkPreprocessor,
)
from pocketshield.network.schema import MODEL_FEATURE_COLUMNS


def make_row(**updates):
    row = {column: "0" for column in MODEL_FEATURE_COLUMNS}
    row.update({"proto": "tcp", "service": "-", "state": "FIN"})
    row.update(updates)
    return row


def column_index(preprocessor, feature_name):
    return preprocessor.get_feature_names_out().index(feature_name)


def test_log1p_is_applied_only_to_selected_nonnegative_skewed_features():
    rows = [make_row(dbytes="0", sttl="0"), make_row(dbytes="1", sttl="1"), make_row(dbytes="3", sttl="3")]
    preprocessor = NetworkPreprocessor(rare_category_min_count=1).fit(rows)

    transformed = preprocessor.transform(rows)
    dbytes_index = column_index(preprocessor, "dbytes")

    assert "dbytes" in LOG1P_FEATURES
    assert "sttl" not in LOG1P_FEATURES
    assert transformed[0][dbytes_index] == pytest.approx(-1.0)
    assert transformed[1][dbytes_index] == pytest.approx(0.0)
    assert transformed[2][dbytes_index] == pytest.approx(1.0)
    assert preprocessor.numeric_impute_values_["sttl"] == 1.0


def test_categorical_features_are_one_hot_encoded_in_a_stable_order():
    rows = [
        make_row(proto="tcp", service="http", state="FIN"),
        make_row(proto="udp", service="dns", state="INT"),
    ]
    preprocessor = NetworkPreprocessor(rare_category_min_count=1).fit(rows)

    transformed = preprocessor.transform(rows)
    names = preprocessor.get_feature_names_out()

    for category_feature in ("proto", "service", "state"):
        encoded_names = [name for name in names if name.startswith(f"{category_feature}=")]
        assert encoded_names
        for row_values in transformed:
            assert sum(row_values[names.index(name)] for name in encoded_names) == 1.0
    assert transformed[0][names.index("proto=tcp")] == 1.0
    assert transformed[1][names.index("proto=udp")] == 1.0


def test_service_dash_is_kept_as_a_regular_category():
    rows = [make_row(service="-"), make_row(service="http")]
    preprocessor = NetworkPreprocessor(rare_category_min_count=1).fit(rows)
    transformed = preprocessor.transform([rows[0]])[0]

    assert "service=-" in preprocessor.get_feature_names_out()
    assert transformed[column_index(preprocessor, "service=-")] == 1.0


def test_rare_and_unseen_categories_get_distinct_explicit_columns():
    rows = [make_row(proto="tcp"), make_row(proto="tcp"), make_row(proto="udp")]
    preprocessor = NetworkPreprocessor(rare_category_min_count=2).fit(rows)
    rare_row, unknown_row = preprocessor.transform(
        [make_row(proto="udp"), make_row(proto="icmp")]
    )
    names = preprocessor.get_feature_names_out()

    assert rare_row[names.index(f"proto={RARE_CATEGORY}")] == 1.0
    assert unknown_row[names.index(f"proto={UNKNOWN_CATEGORY}")] == 1.0
    assert "icmp" not in preprocessor.category_mapping_["proto"]
    assert preprocessor.last_transform_report_["rare_categories"]["proto"] == {"udp": 1}
    assert preprocessor.last_transform_report_["unseen_categories"]["proto"] == {"icmp": 1}


@pytest.mark.parametrize("ftp_value", ["2", "4"])
def test_ftp_login_values_are_preserved_and_reported_as_unusual(ftp_value):
    rows = [make_row(is_ftp_login="0"), make_row(is_ftp_login=ftp_value)]
    preprocessor = NetworkPreprocessor(rare_category_min_count=1).fit(rows)
    transformed = preprocessor.transform(rows)

    ftp_index = column_index(preprocessor, "is_ftp_login")
    assert transformed[1][ftp_index] == float(ftp_value)
    assert preprocessor.fit_report_["unusual_binary_values"]["is_ftp_login"] == {ftp_value: 1}


def test_transform_reuses_fitted_parameters_and_does_not_learn_new_categories():
    training_rows = [make_row(proto="tcp", sbytes="10"), make_row(proto="tcp", sbytes="20")]
    preprocessor = NetworkPreprocessor(rare_category_min_count=1).fit(training_rows)
    fitted_categories = dict(preprocessor.category_mapping_["proto"])
    fitted_medians = dict(preprocessor.numeric_impute_values_)
    output_names = preprocessor.get_feature_names_out()

    transformed = preprocessor.transform([make_row(proto="a-new-proto", sbytes="100000")])

    assert transformed[0][output_names.index(f"proto={UNKNOWN_CATEGORY}")] == 1.0
    assert preprocessor.category_mapping_["proto"] == fitted_categories
    assert preprocessor.numeric_impute_values_ == fitted_medians
    assert preprocessor.get_feature_names_out() == output_names


def test_missing_required_columns_are_rejected():
    preprocessor = NetworkPreprocessor().fit([make_row()])
    incomplete = make_row()
    del incomplete["service"]

    with pytest.raises(ValueError, match="service"):
        preprocessor.transform([incomplete])


def test_invalid_nonfinite_and_negative_log_values_produce_finite_output():
    preprocessor = NetworkPreprocessor().fit([make_row(sbytes="10", dbytes="2")])
    row = make_row(sbytes="nan", dbytes="-4", sttl="-10", dload="inf")

    transformed = preprocessor.transform([row])[0]

    assert all(math.isfinite(value) for value in transformed)
    report = preprocessor.last_transform_report_
    assert report["invalid_numeric_values"]["sbytes"] == 1
    assert report["invalid_numeric_values"]["dbytes"] == 1
    assert report["invalid_numeric_values"]["dload"] == 1
    assert report["negative_values_imputed_for_log1p"]["dbytes"] == 1
    assert report["negative_numeric_values"]["sttl"] == 1


def test_training_invalid_values_are_reported_and_imputed_from_training_data():
    rows = [make_row(sbytes="10", dbytes="2"), make_row(sbytes="nan", dbytes="-4")]
    preprocessor = NetworkPreprocessor().fit(rows)
    transformed = preprocessor.transform(rows)

    assert all(math.isfinite(value) for row in transformed for value in row)
    assert preprocessor.fit_report_["invalid_numeric_values"]["sbytes"] == 1
    assert preprocessor.fit_report_["invalid_numeric_values"]["dbytes"] == 1
    assert preprocessor.fit_report_["negative_values_imputed_for_log1p"]["dbytes"] == 1


def test_output_dimensions_are_reproducible_for_train_validation_and_test_rows():
    training_rows = [
        make_row(proto="tcp", service="-", state="FIN"),
        make_row(proto="tcp", service="http", state="FIN"),
        make_row(proto="udp", service="dns", state="INT"),
    ]
    preprocessor_a = NetworkPreprocessor(rare_category_min_count=2)
    train_a = preprocessor_a.fit_transform(training_rows)
    validation_a = preprocessor_a.transform([make_row(proto="icmp", service="-")])
    test_a = preprocessor_a.transform([make_row(proto="udp", service="unknown-service")])

    preprocessor_b = NetworkPreprocessor(rare_category_min_count=2)
    train_b = preprocessor_b.fit_transform(training_rows)

    dimensions = {len(row) for row in train_a + validation_a + test_a}
    assert len(dimensions) == 1
    assert len(train_a) == 3
    assert train_a == train_b
    assert preprocessor_a.get_feature_names_out() == preprocessor_b.get_feature_names_out()


def test_transform_requires_fit_first_and_fit_rejects_empty_training_data():
    preprocessor = NetworkPreprocessor()

    with pytest.raises(RuntimeError, match="fit"):
        preprocessor.transform([make_row()])
    with pytest.raises(ValueError, match="empty"):
        preprocessor.fit([])
