"""Dispatch inputs to PocketShield's existing local detectors."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from pocketshield.network.anomaly_detector import NetworkAnomalyDetector
from pocketshield.network.schema import select_model_features
from pocketshield.phishing.url_detector import analyze_url as _analyze_url
from pocketshield.sms.sms_detector import analyze_sms as _analyze_sms


class DetectionEngine:
    """Small unified interface for the URL, SMS, and network detectors.

    URL and SMS results are returned unchanged under ``detector_result``.
    Network classification uses a caller-supplied, calibrated threshold; the
    engine does not combine scores from different detectors.
    """

    def __init__(self, network_detector: NetworkAnomalyDetector | None = None) -> None:
        self.network_detector = network_detector

    @staticmethod
    def _response(modality: str, status: str, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "modality": modality,
            "analysis_status": status,
            "detector_result": result,
        }

    def analyze_url(self, url: object) -> dict[str, Any]:
        result = _analyze_url(url)  # type: ignore[arg-type]
        return self._response("url", result["analysis_status"], result)

    def analyze_sms(self, message: object) -> dict[str, Any]:
        result = _analyze_sms(message)  # type: ignore[arg-type]
        return self._response("sms", result["analysis_status"], result)

    def analyze_network_flow(
        self, flow: Mapping[str, object], *, threshold: float | None = None
    ) -> dict[str, Any]:
        if threshold is None or isinstance(threshold, bool):
            return self._response(
                "network",
                "invalid",
                {"error": "An explicit calibrated finite threshold is required."},
            )
        try:
            calibrated_threshold = float(threshold)
        except (TypeError, ValueError, OverflowError):
            calibrated_threshold = math.nan
        if not math.isfinite(calibrated_threshold):
            return self._response(
                "network",
                "invalid",
                {"error": "The calibrated threshold must be a finite number."},
            )
        if not isinstance(flow, Mapping):
            return self._response(
                "network", "invalid", {"error": "A network flow must be a mapping."}
            )
        try:
            select_model_features(flow)
        except ValueError as exc:
            return self._response("network", "invalid", {"error": str(exc)})
        detector = self.network_detector
        if detector is None:
            return self._response(
                "network", "unavailable", {"error": "No network detector is configured."}
            )
        if not getattr(detector, "is_fitted_", False):
            return self._response(
                "network", "unavailable", {"error": "The network detector is not fitted."}
            )

        try:
            score = detector.anomaly_scores([flow])[0]
        except (ValueError, RuntimeError, TypeError) as exc:
            return self._response("network", "invalid", {"error": str(exc)})
        result = {
            "anomaly_score": score,
            "threshold": calibrated_threshold,
            "is_suspicious": score >= calibrated_threshold,
        }
        return self._response("network", "analyzed", result)
