"""Tests for the unified DetectionEngine wrapper."""

import pytest

from pocketshield.engine import DetectionEngine
from pocketshield.network.anomaly_detector import NetworkAnomalyDetector
from pocketshield.network.schema import MODEL_FEATURE_COLUMNS
from pocketshield.phishing.url_detector import analyze_url
from pocketshield.sms.sms_detector import analyze_sms


def make_flow(**updates):
    flow = {column: "0" for column in MODEL_FEATURE_COLUMNS}
    flow.update({"proto": "tcp", "service": "-", "state": "FIN", "dbytes": "10"})
    flow.update(updates)
    return flow


class StubNetworkDetector:
    is_fitted_ = True

    def __init__(self, score=0.6):
        self.score = score

    def anomaly_scores(self, rows):
        assert len(rows) == 1
        return [self.score]


def test_url_dispatch_preserves_detector_result():
    engine = DetectionEngine()
    response = engine.analyze_url("https://example.com")

    assert response == {
        "modality": "url",
        "analysis_status": "analyzed",
        "detector_result": analyze_url("https://example.com"),
    }


def test_sms_dispatch_preserves_detector_result():
    engine = DetectionEngine()
    response = engine.analyze_sms("Your package is arriving today.")

    assert response == {
        "modality": "sms",
        "analysis_status": "analyzed",
        "detector_result": analyze_sms("Your package is arriving today."),
    }


@pytest.mark.parametrize(("threshold", "expected"), [(0.6, True), (0.61, False)])
def test_network_scoring_uses_explicit_inclusive_threshold(threshold, expected):
    response = DetectionEngine(StubNetworkDetector()).analyze_network_flow(
        make_flow(), threshold=threshold
    )

    assert response == {
        "modality": "network",
        "analysis_status": "analyzed",
        "detector_result": {
            "anomaly_score": 0.6,
            "threshold": threshold,
            "is_suspicious": expected,
        },
    }


@pytest.mark.parametrize("value", [None, "", 4, object()])
def test_url_and_sms_invalid_inputs_keep_detector_status(value):
    engine = DetectionEngine()

    assert engine.analyze_url(value)["analysis_status"] == "invalid"
    assert engine.analyze_sms(value)["analysis_status"] == "invalid"


def test_network_requires_explicit_finite_threshold_and_mapping_input():
    engine = DetectionEngine(StubNetworkDetector())

    assert engine.analyze_network_flow(make_flow())["analysis_status"] == "invalid"
    assert engine.analyze_network_flow(make_flow(), threshold=float("nan"))["analysis_status"] == "invalid"
    assert engine.analyze_network_flow([], threshold=0.5)["analysis_status"] == "invalid"


def test_network_handles_missing_and_unfitted_detector():
    missing = DetectionEngine().analyze_network_flow(make_flow(), threshold=0.5)
    unfitted = DetectionEngine(NetworkAnomalyDetector()).analyze_network_flow(
        make_flow(), threshold=0.5
    )

    assert missing["analysis_status"] == "unavailable"
    assert unfitted["analysis_status"] == "unavailable"


def test_network_handles_detector_input_errors_gracefully():
    response = DetectionEngine(StubNetworkDetector()).analyze_network_flow(
        {"incomplete": "row"}, threshold=0.5
    )

    assert response["analysis_status"] == "invalid"
    assert "error" in response["detector_result"]
