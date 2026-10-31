"""UNSW-NB15 CSV and model-input schema definitions.

This module describes column names and groups only. It deliberately does not
convert values: for example, ``is_ftp_login`` values such as ``"2"`` and
``"4"`` and the ``service`` value ``"-"`` remain unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


ID_COLUMN = "id"
LABEL_COLUMN = "label"
ATTACK_CATEGORY_COLUMN = "attack_cat"
TARGET_COLUMNS = (ATTACK_CATEGORY_COLUMN, LABEL_COLUMN)

# Exact model input columns, in the order used by the supplied CSV files,
# excluding the row identifier and both target columns.
MODEL_FEATURE_COLUMNS = (
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
)

# Alias with a short generic name for callers that already know this is the
# model's input feature list.
FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS

CATEGORICAL_FEATURES = (
    "proto",
    "service",
    "state",
)

NUMERICAL_FEATURES = (
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
)

# The feature metadata describes these as binary. Keep is_ftp_login in this
# group even though the supplied files also contain 2 and 4; schema handling
# must not coerce those observed values to True or False.
BINARY_FEATURES = (
    "is_ftp_login",
    "is_sm_ips_ports",
)

# Full CSV schema, including non-model columns, in source-file order.
CSV_COLUMNS = (
    ID_COLUMN,
    *MODEL_FEATURE_COLUMNS,
    ATTACK_CATEGORY_COLUMN,
    LABEL_COLUMN,
)


@dataclass(frozen=True)
class CSVSchemaValidation:
    """Result of comparing CSV header names with the UNSW-NB15 schema."""

    missing_columns: tuple[str, ...]
    unexpected_columns: tuple[str, ...]
    duplicate_columns: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Whether the header has all expected names and no extras/duplicates."""
        return not (self.missing_columns or self.unexpected_columns or self.duplicate_columns)


def validate_csv_schema(columns: Iterable[str]) -> CSVSchemaValidation:
    """Report required, unexpected, and duplicated CSV header columns.

    Column order is not required: a CSV reader that addresses fields by name
    can safely consume a reordered header. Values and their types are not
    inspected or converted here.
    """
    observed = tuple(columns)
    expected_set = set(CSV_COLUMNS)
    observed_set = set(observed)

    seen: set[str] = set()
    duplicate_columns: list[str] = []
    for column in observed:
        if column in seen and column not in duplicate_columns:
            duplicate_columns.append(column)
        seen.add(column)

    return CSVSchemaValidation(
        missing_columns=tuple(column for column in CSV_COLUMNS if column not in observed_set),
        unexpected_columns=tuple(column for column in observed if column not in expected_set),
        duplicate_columns=tuple(duplicate_columns),
    )


def select_model_features(row: Mapping[str, object]) -> dict[str, object]:
    """Select the 42 feature values without converting or altering them.

    Extra row fields such as ``id``, ``label``, and ``attack_cat`` are ignored.
    Missing model-feature fields raise ``ValueError``.
    """
    missing = tuple(column for column in MODEL_FEATURE_COLUMNS if column not in row)
    if missing:
        raise ValueError(f"Row is missing model feature columns: {', '.join(missing)}")
    return {column: row[column] for column in MODEL_FEATURE_COLUMNS}

