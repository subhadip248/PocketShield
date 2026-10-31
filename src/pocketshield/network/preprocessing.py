"""Fit-on-training-only preprocessing for UNSW-NB15 flow features.

The implementation uses only the Python standard library. It returns
row-major ``list[list[float]]`` data suitable for a later model, but does not
train or invoke a model itself.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import math
from statistics import median

from pocketshield.network.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    select_model_features,
)


# These nonnegative measurements are commonly right-skewed. Bounded protocol
# fields, TCP sequence numbers/windows, and binary fields are intentionally
# excluded from log1p.
LOG1P_FEATURES = (
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "tcprtt",
    "synack",
    "ackdat",
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

RARE_CATEGORY = "__RARE__"
UNKNOWN_CATEGORY = "__UNKNOWN__"
MISSING_CATEGORY = "__MISSING__"
_RESERVED_CATEGORIES = {RARE_CATEGORY, UNKNOWN_CATEGORY, MISSING_CATEGORY}
_MAX_ABS_SCALED_VALUE = 1e12


def _as_finite_float(value: object) -> float | None:
    """Parse a numeric cell; return None for missing, invalid, or non-finite data."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _category_value(value: object) -> str:
    """Normalize a category without treating the dataset's '-' as missing."""
    if value is None:
        return MISSING_CATEGORY
    normalized = str(value).strip()
    return normalized if normalized else MISSING_CATEGORY


def _quantile(sorted_values: list[float], fraction: float) -> float:
    """Compute a linearly interpolated quantile from already-sorted values."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    portion = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * portion


class NetworkPreprocessor:
    """Prepare UNSW-NB15 rows using parameters learned only by :meth:`fit`.

    ``fit`` and ``transform`` accept iterables of mappings, such as rows from
    ``csv.DictReader``. The transformed columns have a stable order exposed by
    ``feature_names_out_``. Numeric missing/invalid cells and negative values
    for log1p features are imputed with the training median. Binary numeric
    values are passed through unchanged (apart from conversion to float), so
    unusual FTP values such as 2 and 4 are preserved and reported.
    """

    def __init__(self, rare_category_min_count: int = 2) -> None:
        if not isinstance(rare_category_min_count, int) or rare_category_min_count < 1:
            raise ValueError("rare_category_min_count must be a positive integer")
        self.rare_category_min_count = rare_category_min_count
        self.is_fitted_ = False

    def fit(self, rows: Iterable[Mapping[str, object]]) -> NetworkPreprocessor:
        """Learn medians, robust-scaling statistics, and category vocabularies."""
        materialized = [select_model_features(row) for row in rows]
        if not materialized:
            raise ValueError("Cannot fit preprocessing on an empty dataset")

        numeric_training: dict[str, list[float]] = {name: [] for name in NUMERICAL_FEATURES}
        invalid_numeric_counts: Counter[str] = Counter()
        negative_numeric_counts: Counter[str] = Counter()
        negative_log_counts: Counter[str] = Counter()
        binary_training_values: dict[str, list[float]] = {name: [] for name in BINARY_FEATURES}
        unusual_binary_counts: dict[str, Counter[str]] = {
            name: Counter() for name in BINARY_FEATURES
        }
        category_counts: dict[str, Counter[str]] = {
            name: Counter() for name in CATEGORICAL_FEATURES
        }

        for row in materialized:
            for name in NUMERICAL_FEATURES + BINARY_FEATURES:
                raw_value = row[name]
                value = _as_finite_float(raw_value)
                if value is None:
                    invalid_numeric_counts[name] += 1
                    continue
                if value < 0:
                    negative_numeric_counts[name] += 1
                if name in BINARY_FEATURES:
                    binary_training_values[name].append(value)
                    if value not in (0.0, 1.0):
                        unusual_binary_counts[name][str(raw_value).strip()] += 1
                    continue
                if name in LOG1P_FEATURES:
                    if value < 0:
                        negative_log_counts[name] += 1
                        invalid_numeric_counts[name] += 1
                        continue
                    value = math.log1p(value)
                numeric_training[name].append(value)

            for name in CATEGORICAL_FEATURES:
                category_counts[name][_category_value(row[name])] += 1

        self.numeric_impute_values_: dict[str, float] = {}
        self.numeric_centers_: dict[str, float] = {}
        self.numeric_scales_: dict[str, float] = {}
        for name in NUMERICAL_FEATURES:
            values = numeric_training[name]
            fill_value = median(values) if values else 0.0
            self.numeric_impute_values_[name] = fill_value
            completed = sorted(values + [fill_value] * (len(materialized) - len(values)))
            center = median(completed) if completed else 0.0
            scale = _quantile(completed, 0.75) - _quantile(completed, 0.25)
            self.numeric_centers_[name] = center
            self.numeric_scales_[name] = scale if scale > 0 and math.isfinite(scale) else 1.0
        self._binary_training_values_ = binary_training_values

        self.category_mapping_: dict[str, dict[str, str]] = {}
        self.category_levels_: dict[str, tuple[str, ...]] = {}
        rare_categories: dict[str, tuple[str, ...]] = {}
        for name in CATEGORICAL_FEATURES:
            mapping: dict[str, str] = {}
            rare: list[str] = []
            for value, count in category_counts[name].items():
                if value == MISSING_CATEGORY:
                    mapping[value] = MISSING_CATEGORY
                    continue
                # Escape a literal reserved token so it cannot collide with an
                # internally generated rare, unknown, or missing category.
                encoded_value = f"__VALUE__:{value}" if value in _RESERVED_CATEGORIES else value
                if count < self.rare_category_min_count:
                    mapping[value] = RARE_CATEGORY
                    rare.append(value)
                else:
                    mapping[value] = encoded_value
            self.category_mapping_[name] = mapping
            self.category_levels_[name] = tuple(
                sorted({value for value in mapping.values() if value not in _RESERVED_CATEGORIES})
                + [RARE_CATEGORY, UNKNOWN_CATEGORY, MISSING_CATEGORY]
            )
            rare_categories[name] = tuple(sorted(rare))

        self.feature_names_out_ = self._build_feature_names()
        self.fit_report_ = {
            "invalid_numeric_values": dict(invalid_numeric_counts),
            "negative_numeric_values": dict(negative_numeric_counts),
            "negative_values_imputed_for_log1p": dict(negative_log_counts),
            "unusual_binary_values": {
                name: dict(counts) for name, counts in unusual_binary_counts.items() if counts
            },
            "rare_categories": rare_categories,
        }
        self.is_fitted_ = True
        return self

    def transform(self, rows: Iterable[Mapping[str, object]]) -> list[list[float]]:
        """Transform rows using fitted parameters without learning from them."""
        self._require_fitted()
        output: list[list[float]] = []
        invalid_numeric_counts: Counter[str] = Counter()
        negative_numeric_counts: Counter[str] = Counter()
        negative_log_counts: Counter[str] = Counter()
        unusual_binary_counts: dict[str, Counter[str]] = {
            name: Counter() for name in BINARY_FEATURES
        }
        rare_category_counts: dict[str, Counter[str]] = {
            name: Counter() for name in CATEGORICAL_FEATURES
        }
        unknown_category_counts: dict[str, Counter[str]] = {
            name: Counter() for name in CATEGORICAL_FEATURES
        }
        for source_row in rows:
            row = select_model_features(source_row)
            transformed: list[float] = []
            for name in MODEL_FEATURE_COLUMNS:
                if name in CATEGORICAL_FEATURES:
                    value = _category_value(row[name])
                    mapped = self.category_mapping_[name].get(value)
                    if mapped is None:
                        mapped = UNKNOWN_CATEGORY
                        unknown_category_counts[name][value] += 1
                    elif mapped == RARE_CATEGORY:
                        rare_category_counts[name][value] += 1
                    level_values = self.category_levels_[name]
                    transformed.extend(1.0 if level == mapped else 0.0 for level in level_values)
                elif name in BINARY_FEATURES:
                    value = _as_finite_float(row[name])
                    if value is None:
                        invalid_numeric_counts[name] += 1
                        value = self._binary_impute_value(name)
                    elif value not in (0.0, 1.0):
                        unusual_binary_counts[name][str(row[name]).strip()] += 1
                    transformed.append(value)
                else:
                    raw_numeric = _as_finite_float(row[name])
                    if raw_numeric is None:
                        invalid_numeric_counts[name] += 1
                        value = self.numeric_impute_values_[name]
                    elif raw_numeric < 0:
                        negative_numeric_counts[name] += 1
                        if name in LOG1P_FEATURES:
                            negative_log_counts[name] += 1
                            invalid_numeric_counts[name] += 1
                            value = self.numeric_impute_values_[name]
                        else:
                            value = raw_numeric
                    elif name in LOG1P_FEATURES:
                        value = math.log1p(raw_numeric)
                    else:
                        value = raw_numeric
                    value = self._robust_scale(value, name)
                    transformed.append(value)

            if not all(math.isfinite(value) for value in transformed):
                raise RuntimeError("Preprocessing produced a non-finite output value")
            output.append(transformed)
        self.last_transform_report_ = {
            "invalid_numeric_values": dict(invalid_numeric_counts),
            "negative_numeric_values": dict(negative_numeric_counts),
            "negative_values_imputed_for_log1p": dict(negative_log_counts),
            "unusual_binary_values": {
                name: dict(counts) for name, counts in unusual_binary_counts.items() if counts
            },
            "rare_categories": {
                name: dict(counts) for name, counts in rare_category_counts.items() if counts
            },
            "unseen_categories": {
                name: dict(counts) for name, counts in unknown_category_counts.items() if counts
            },
        }
        return output

    def fit_transform(self, rows: Iterable[Mapping[str, object]]) -> list[list[float]]:
        """Fit on rows and return their transformed representation."""
        materialized = list(rows)
        self.fit(materialized)
        return self.transform(materialized)

    def get_feature_names_out(self) -> tuple[str, ...]:
        """Return the stable ordered names corresponding to transformed columns."""
        self._require_fitted()
        return self.feature_names_out_

    def _binary_impute_value(self, name: str) -> float:
        """Use the training median for a missing/invalid binary cell."""
        # Binary cells are deliberately not scaled. The median is learned from
        # fit data only; unusual finite values remain unchanged.
        values = self._binary_training_values_.get(name, [])
        return median(values) if values else 0.0

    def _robust_scale(self, value: float, name: str) -> float:
        scaled = (value - self.numeric_centers_[name]) / self.numeric_scales_[name]
        if not math.isfinite(scaled):
            return math.copysign(_MAX_ABS_SCALED_VALUE, scaled)
        return max(-_MAX_ABS_SCALED_VALUE, min(_MAX_ABS_SCALED_VALUE, scaled))

    def _build_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for name in MODEL_FEATURE_COLUMNS:
            if name in CATEGORICAL_FEATURES:
                names.extend(f"{name}={level}" for level in self.category_levels_[name])
            else:
                names.append(name)
        return tuple(names)

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before transform()")

