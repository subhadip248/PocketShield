"""Manually test a detector-grounded explanation with the real ExecuTorch model.

Run from the repository root with:
    PYTHONPATH=src ./.venv-executorch/bin/python scripts/explanation_smoke.py
"""

import json

from pocketshield.engine import DetectionEngine
from pocketshield.explanation import AIExplanationEngine


def main() -> None:
    detection = DetectionEngine().analyze_url("http://192.0.2.1/login")
    explanation = AIExplanationEngine().explain(detection)
    print(json.dumps(explanation, indent=2))
    if explanation.get("explanation_source") != "model_selected_finding":
        raise RuntimeError("The real-model explanation smoke test did not select a finding")


if __name__ == "__main__":
    main()
