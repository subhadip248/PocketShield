"""Run a real-model inference smoke test with the exported PocketShield Llama.

From the repository root:
    ./.venv-executorch/bin/python scripts/llama3_2_1b_smoke.py
"""

from pathlib import Path

# Register the operators required by the exported model before loading it.
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
from executorch.kernels import quantized  # noqa: F401


MODEL_PATH = Path("outputs/llama3_2_1b/llama3_2.pte")
TOKENIZER_PATH = Path(
    "/mnt/d/PocketShield-Models/llama-3.2-1b-instruct/original/tokenizer.model"
)
PROMPT = "Explain phishing in one sentence."


def main() -> None:
    for path in (MODEL_PATH, TOKENIZER_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required inference artifact not found: {path}")

    runner = TextLLMRunner(str(MODEL_PATH), str(TOKENIZER_PATH))
    output: list[str] = []
    runner.generate(
        PROMPT,
        GenerationConfig(echo=False, max_new_tokens=33, temperature=0.0),
        lambda token: output.append(token),
    )

    response = "".join(output).strip()
    if not response:
        raise RuntimeError("TextLLMRunner completed without generating response text")
    print(response)


if __name__ == "__main__":
    main()
