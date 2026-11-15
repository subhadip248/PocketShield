"""Tests for grounded explanations using a lightweight mocked runner."""

from pocketshield.explanation import AIExplanationEngine


class MockRunner:
    def __init__(self, response="1"):
        self.response = response
        self.calls = []

    def generate(self, prompt, config, callback):
        self.calls.append((prompt, config))
        callback(self.response)


def url_detection():
    return {
        "modality": "url",
        "analysis_status": "analyzed",
        "detector_result": {
            "original_url": "http://untrusted.example/login?secret=private",
            "normalized_url": "http://untrusted.example/login?secret=private",
            "analysis_status": "analyzed",
            "risk_score": 45,
            "risk_level": "Medium",
            "is_suspicious": True,
            "detected_indicators": [
                "Uses HTTP without transport encryption.",
                "Hostname has 4 subdomains, more than the threshold of 3.",
            ],
            "explanation": "Detector explanation.",
        },
    }


def test_url_explanation_uses_mock_runner_and_preserves_detector_decision():
    runner = MockRunner()
    engine = AIExplanationEngine(runner=runner, config_factory=lambda: "mock-config")
    detection = url_detection()

    result = engine.explain(detection)

    assert result["analysis_status"] == "explained"
    assert result["explanation_source"] == "model_selected_finding"
    assert result["threat_explanation"] == (
        "The detector marked this url result suspicious because it found: "
        "Uses HTTP without transport encryption."
    )
    assert result["reasons"] == detection["detector_result"]["detected_indicators"]
    assert result["detector_result"] == detection["detector_result"]
    assert result["detector_result"]["risk_score"] == 45
    assert result["detector_result"]["is_suspicious"] is True
    prompt = runner.calls[0][0]
    assert "Uses HTTP without transport encryption." in prompt
    assert "untrusted.example" not in prompt
    assert "secret=private" not in prompt


def test_runner_is_created_once_and_reused_for_multiple_explanations():
    runner = MockRunner()
    factory_calls = []

    def factory(model_path, tokenizer_path):
        factory_calls.append((model_path, tokenizer_path))
        return runner

    engine = AIExplanationEngine(
        runner_factory=factory,
        config_factory=lambda: "mock-config",
    )
    engine.explain(url_detection())
    engine.explain(url_detection())

    assert len(factory_calls) == 1
    assert len(runner.calls) == 2


def test_sms_uses_only_detector_reasons_and_fixed_recommendations():
    runner = MockRunner("2")
    engine = AIExplanationEngine(runner=runner, config_factory=lambda: "mock-config")
    detection = {
        "modality": "sms",
        "analysis_status": "analyzed",
        "detector_result": {
            "analysis_status": "analyzed",
            "risk_score": 70,
            "risk_level": "High",
            "is_suspicious": True,
            "reasons": ["The message asks for an OTP.", "The message contains a risky link."],
        },
    }

    result = engine.explain(detection)

    assert result["reasons"] == detection["detector_result"]["reasons"]
    assert "verification codes" in result["recommendations"][0]
    assert "OTP" in runner.calls[0][0]
    assert result["explanation_source"] == "model_selected_finding"


def test_network_reason_states_only_the_score_threshold_comparison():
    runner = MockRunner("1")
    engine = AIExplanationEngine(runner=runner, config_factory=lambda: "mock-config")
    result = engine.explain(
        {
            "modality": "network",
            "analysis_status": "analyzed",
            "detector_result": {
                "anomaly_score": 0.7,
                "threshold": 0.5,
                "is_suspicious": True,
            },
        }
    )

    assert result["reasons"] == ["Anomaly score 0.7 is at or above threshold 0.5."]
    assert "source IP" not in runner.calls[0][0]
    assert result["detector_result"]["is_suspicious"] is True
    assert result["explanation_source"] == "model_selected_finding"


def test_invalid_detection_is_rejected_without_loading_runner():
    factory_calls = []
    engine = AIExplanationEngine(
        runner_factory=lambda *_: factory_calls.append(True),
        config_factory=lambda: "mock-config",
    )

    result = engine.explain({"modality": "url", "analysis_status": "invalid", "detector_result": {}})

    assert result["analysis_status"] == "invalid"
    assert result["error"] == "unsupported_detection_result"
    assert factory_calls == []


def test_inference_failure_returns_safe_fallback_without_internal_details():
    class FailingRunner:
        def generate(self, *_args):
            raise RuntimeError("/private/path/model.pte failed")

    result = AIExplanationEngine(
        runner=FailingRunner(), config_factory=lambda: "mock-config"
    ).explain(url_detection())

    assert result["analysis_status"] == "explained"
    assert result["explanation_source"] == "fallback"
    assert result["threat_explanation"] == (
        "The detector marked this url result suspicious because it found: "
        "Uses HTTP without transport encryption."
    )
    assert "/private" not in str(result)


def test_ungrounded_model_text_uses_detector_derived_fallback():
    runner = MockRunner("The IP address proves criminal intent.")

    result = AIExplanationEngine(
        runner=runner, config_factory=lambda: "mock-config"
    ).explain(url_detection())

    assert result["explanation_source"] == "fallback"
    assert "criminal" not in result["threat_explanation"]
