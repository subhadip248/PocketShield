"""Tests for the UNSW-NB15 dataset schema definitions."""

import pytest

from pocketshield.network.schema import (
    ATTACK_CATEGORY_COLUMN,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CSV_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    LABEL_COLUMN,
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMNS,
    select_model_features,
    validate_csv_schema,
)


def test_model_schema_defines_exactly_42_features_in_csv_order():
    assert len(MODEL_FEATURE_COLUMNS) == 42
    assert len(set(MODEL_FEATURE_COLUMNS)) == 42
    assert FEATURE_COLUMNS == MODEL_FEATURE_COLUMNS
    assert CSV_COLUMNS == (
        ID_COLUMN,
        *MODEL_FEATURE_COLUMNS,
        ATTACK_CATEGORY_COLUMN,
        LABEL_COLUMN,
    )


def test_model_features_exclude_identifier_and_targets():
    assert ID_COLUMN not in MODEL_FEATURE_COLUMNS
    assert all(target not in MODEL_FEATURE_COLUMNS for target in TARGET_COLUMNS)
    assert TARGET_COLUMNS == (ATTACK_CATEGORY_COLUMN, LABEL_COLUMN)


def test_feature_groups_are_complete_and_disjoint():
    groups = set(NUMERICAL_FEATURES) | set(CATEGORICAL_FEATURES) | set(BINARY_FEATURES)

    assert len(NUMERICAL_FEATURES) == 37
    assert len(CATEGORICAL_FEATURES) == 3
    assert len(BINARY_FEATURES) == 2
    assert groups == set(MODEL_FEATURE_COLUMNS)
    assert len(groups) == len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES) + len(BINARY_FEATURES)
    assert CATEGORICAL_FEATURES == ("proto", "service", "state")
    assert "is_ftp_login" in BINARY_FEATURES
    assert "is_sm_ips_ports" in BINARY_FEATURES


def test_csv_schema_reports_missing_and_unexpected_columns():
    header = list(CSV_COLUMNS)
    header.remove("state")
    header.append("unexpected_feature")

    result = validate_csv_schema(header)

    assert result.missing_columns == ("state",)
    assert result.unexpected_columns == ("unexpected_feature",)
    assert result.duplicate_columns == ()
    assert result.is_valid is False


def test_csv_schema_accepts_reordered_columns_and_reports_duplicates():
    result = validate_csv_schema((*reversed(CSV_COLUMNS), "proto"))

    assert result.missing_columns == ()
    assert result.unexpected_columns == ()
    assert result.duplicate_columns == ("proto",)
    assert result.is_valid is False
    assert validate_csv_schema(reversed(CSV_COLUMNS)).is_valid is True


@pytest.mark.parametrize("ftp_value", ["2", "4"])
def test_feature_selection_preserves_unusual_ftp_values_and_service_placeholder(ftp_value):
    row = {column: "0" for column in MODEL_FEATURE_COLUMNS}
    row.update(
        {
            "service": "-",
            "is_ftp_login": ftp_value,
            "id": "17",
            "label": "1",
            "attack_cat": "Exploits",
        }
    )

    selected = select_model_features(row)

    assert selected["is_ftp_login"] == ftp_value
    assert isinstance(selected["is_ftp_login"], str)
    assert selected["service"] == "-"
    assert "id" not in selected
    assert "label" not in selected
    assert "attack_cat" not in selected


def test_feature_selection_reports_missing_feature_columns():
    with pytest.raises(ValueError, match="service"):
        select_model_features({"dur": "0.5"})
