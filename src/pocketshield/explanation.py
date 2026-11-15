"""Grounded, optional LLM explanations for PocketShield detector results."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from pathlib import Path
import re
import threading
from typing import Any


DEFAULT_MODEL_PATH = Path("outputs/llama3_2_1b/llama3_2.pte")
DEFAULT_TOKENIZER_PATH = Path(
    "/mnt/d/PocketShield-Models/llama-3.2-1b-instruct/original/tokenizer.model"
)
MAX_EXPLANATION_TOKENS = 8


class AIExplanationEngine:
    """Explain structured detector output without changing its assessment.

    The ExecuTorch runner is created lazily and cached for reuse. Pass a
    ``runner`` or factories for unit testing without loading model weights.
    Only detector indicators and numerical assessment fields are included in
    the prompt; original URLs, SMS bodies, and network row values are omitted.
    """

    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
        runner: object | None = None,
        runner_factory: Callable[[str, str], object] | None = None,
        config_factory: Callable[[], object] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self._runner = runner
        self._runner_factory = runner_factory or self._create_runner
        self._config_factory = config_factory or self._create_generation_config
        self._lock = threading.Lock()

    @staticmethod
    def _create_runner(model_path: str, tokenizer_path: str) -> object:
        # These imports register the exported model's operators before load.
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
        from executorch.extension.llm.runner import TextLLMRunner
        from executorch.kernels import quantized  # noqa: F401

        return TextLLMRunner(model_path, tokenizer_path)

    @staticmethod
    def _create_generation_config() -> object:
        from executorch.extension.llm.runner import GenerationConfig

        return GenerationConfig(
            echo=False,
            max_new_tokens=MAX_EXPLANATION_TOKENS,
            temperature=0.0,
        )

    @staticmethod
    def _response(
        *,
        modality: str | None,
        status: str,
        explanation: str,
        reasons: list[str] | None = None,
        recommendations: list[str] | None = None,
        detector_result: Mapping[str, Any] | None = None,
        source: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "modality": modality,
            "analysis_status": status,
            "threat_explanation": explanation,
            "reasons": reasons or [],
            "recommendations": recommendations or [],
        }
        if detector_result is not None:
            response["detector_result"] = dict(detector_result)
        if source is not None:
            response["explanation_source"] = source
        if error is not None:
            response["error"] = error
        return response

    @staticmethod
    def _string_list(value: object) -> list[str] | None:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            return None
        return [item[:240] for item in value[:8]]

    def _extract_evidence(
        self, detection: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any], bool, list[str], dict[str, Any]] | None:
        modality = detection.get("modality")
        detector_result = detection.get("detector_result")
        if (
            not isinstance(modality, str)
            or modality not in {"url", "sms", "network"}
            or not isinstance(detector_result, Mapping)
        ):
            return None
        if detection.get("analysis_status") != "analyzed" or detector_result.get("analysis_status", "analyzed") != "analyzed":
            return None

        suspicious = detector_result.get("is_suspicious")
        if not isinstance(suspicious, bool):
            return None

        facts: dict[str, Any] = {
            "modality": modality,
            "is_suspicious": suspicious,
        }
        reasons: list[str]
        if modality == "url":
            parsed_reasons = self._string_list(detector_result.get("detected_indicators"))
            score = detector_result.get("risk_score")
            level = detector_result.get("risk_level")
            if (
                parsed_reasons is None
                or not isinstance(score, int)
                or isinstance(score, bool)
                or not isinstance(level, str)
            ):
                return None
            reasons = parsed_reasons
            facts.update({"risk_score": score, "risk_level": level, "indicators": reasons})
            if not reasons:
                reasons = ["No configured URL risk indicators were found."]
                facts["indicators"] = reasons
        elif modality == "sms":
            parsed_reasons = self._string_list(detector_result.get("reasons"))
            score = detector_result.get("risk_score")
            level = detector_result.get("risk_level")
            if parsed_reasons is None or not isinstance(score, int) or isinstance(score, bool) or not isinstance(level, str):
                return None
            reasons = parsed_reasons
            facts.update({"risk_score": score, "risk_level": level, "reasons": reasons})
        else:
            score = detector_result.get("anomaly_score")
            threshold = detector_result.get("threshold")
            if isinstance(score, bool) or isinstance(threshold, bool):
                return None
            try:
                score = float(score)
                threshold = float(threshold)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(score) or not math.isfinite(threshold):
                return None
            facts.update({"anomaly_score": score, "threshold": threshold})
            comparator = "at or above" if suspicious else "below"
            reasons = [f"Anomaly score {score:g} is {comparator} threshold {threshold:g}."]
            facts["threshold_comparison"] = reasons[0]
        return modality, dict(detector_result), suspicious, reasons, facts

    @staticmethod
    def _recommendations(modality: str, suspicious: bool) -> list[str]:
        if not suspicious:
            return ["Continue using normal security precautions; a low detector score does not prove safety."]
        if modality == "url":
            return ["Avoid opening the link; reach the intended service through a trusted route."]
        if modality == "sms":
            return ["Do not reply, follow links, or share passwords or verification codes; verify through a trusted channel."]
        return ["Review the flow against expected network activity and investigate it using trusted monitoring tools."]

    def _get_runner(self) -> object:
        if self._runner is None:
            self._runner = self._runner_factory(str(self.model_path), str(self.tokenizer_path))
        return self._runner

    def _generate(self, facts: dict[str, Any], reasons: list[str]) -> int:
        numbered_reasons = "\n".join(
            f"{index}. {reason}" for index, reason in enumerate(reasons, start=1)
        )
        prompt = (
            "<|start_header_id|>system<|end_header_id|>Select the one listed finding that best "
            "explains the existing detector result. Do not classify or add facts. Reply with only "
            "its number.\n<|eot_id|><|start_header_id|>user<|end_header_id|>"
            f"Modality: {facts['modality']}; detector suspicious flag: {facts['is_suspicious']}.\n"
            f"Findings:\n{numbered_reasons}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        generated: list[str] = []
        runner = self._get_runner()
        runner.generate(
            prompt,
            self._config_factory(),
            lambda token: generated.append(str(token)),
        )
        selection = re.sub(r"<\|[^|]+\|>", "", "".join(generated)).strip()
        match = re.match(r"^(\d+)(?=[.)\s]|$)", selection)
        if match is None:
            raise RuntimeError("Invalid finding selection")
        selected = int(match.group(1))
        if not 1 <= selected <= len(reasons):
            raise RuntimeError("Finding selection is out of range")
        return selected - 1

    def explain(self, detection: Mapping[str, Any]) -> dict[str, Any]:
        """Return a grounded explanation for one analyzed engine response."""
        if not isinstance(detection, Mapping):
            return self._response(
                modality=None,
                status="invalid",
                explanation="A structured detector result is required.",
                error="invalid_detection_result",
            )
        parsed = self._extract_evidence(detection)
        if parsed is None:
            return self._response(
                modality=detection.get("modality") if isinstance(detection.get("modality"), str) else None,
                status="invalid",
                explanation="An analyzed URL, SMS, or network detector result is required.",
                error="unsupported_detection_result",
            )

        modality, detector_result, suspicious, reasons, facts = parsed
        recommendations = self._recommendations(modality, suspicious)
        try:
            with self._lock:
                selected_reason = self._generate(facts, reasons)
            explanation = self._format_explanation(modality, suspicious, reasons[selected_reason])
            source = "model_selected_finding"
        except Exception:
            # Avoid leaking paths, runtime internals, or partial model output.
            selected_reason = reasons[0] if reasons else ""
            explanation = self._format_explanation(modality, suspicious, selected_reason)
            source = "fallback"
        return self._response(
            modality=modality,
            status="explained",
            explanation=explanation,
            reasons=reasons,
            recommendations=recommendations,
            detector_result=detector_result,
            source=source,
        )

    @staticmethod
    def _format_explanation(modality: str, suspicious: bool, reason: str) -> str:
        if not suspicious:
            return f"The detector did not classify this {modality} result as suspicious."
        if reason:
            return f"The detector marked this {modality} result suspicious because it found: {reason}"
        return f"The detector marked this {modality} result suspicious using its configured checks."
