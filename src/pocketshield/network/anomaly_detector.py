from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

from sklearn.ensemble import IsolationForest

from pocketshield.network.preprocessing import NetworkPreprocessor
from pocketshield.network.schema import LABEL_COLUMN, select_model_features


def _parse_label(value: object) -> int:
    """Accept only explicit binary UNSW-NB15 labels (0 normal, 1 attack)."""
    if isinstance(value, bool):
        raise ValueError(f"Invalid label {value!r}; expected 0 (normal) or 1 (attack)")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid label {value!r}; expected 0 (normal) or 1 (attack)") from exc
    if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(f"Invalid label {value!r}; expected 0 (normal) or 1 (attack)")
    return int(numeric)


class NetworkAnomalyDetector:
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        random_state: int = 42,
        rare_category_min_count: int = 2,
    ) -> None:
        if not isinstance(n_estimators, int) or isinstance(n_estimators, bool) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise ValueError("random_state must be an integer")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.preprocessor = NetworkPreprocessor(
            rare_category_min_count=rare_category_min_count
        )
        self.is_fitted_ = False

    def fit(self, rows: Iterable[Mapping[str, object]]) -> NetworkAnomalyDetector:
        """Fit preprocessing and the forest using only rows labeled normal."""
        materialized = list(rows)
        if not materialized:
            raise ValueError("Cannot fit anomaly detector on an empty dataset")

        normal_features: list[dict[str, object]] = []
        attack_count = 0
        for row in materialized:
            if not isinstance(row, Mapping):
                raise ValueError("Each training row must be a mapping of column names to values")
            if LABEL_COLUMN not in row:
                raise ValueError(f"Training row is missing required column: {LABEL_COLUMN}")
            label = _parse_label(row[LABEL_COLUMN])
            # Validate required input columns even on attacks that will not be fitted.
            features = select_model_features(row)
            if label == 0:
                normal_features.append(features)
            else:
                attack_count += 1

        if not normal_features:
            raise ValueError("Training data contains no normal records (label == 0)")

        self.preprocessor.fit(normal_features)
        transformed = self.preprocessor.transform(normal_features)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model_.fit(transformed)
        self.n_normal_training_samples_ = len(normal_features)
        self.n_attack_rows_excluded_ = attack_count
        self.is_fitted_ = True
        return self

    def anomaly_scores(self, rows: Iterable[Mapping[str, object]]) -> list[float]:
        """Return one finite score per row, with higher values more anomalous.

        scikit-learn's ``IsolationForest.score_samples`` has the opposite
        direction (lower means more anomalous), so its values are negated here.
        """
        self._require_fitted()
        materialized = list(rows)
        if not materialized:
            raise ValueError("Cannot score an empty dataset")
        transformed = self.preprocessor.transform(materialized)
        scores = [-float(value) for value in self.model_.score_samples(transformed)]
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError("Isolation Forest produced a non-finite anomaly score")
        return scores

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before anomaly_scores()")

