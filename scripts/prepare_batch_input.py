"""
Convert processed QuALITY datasets into OpenAI Batch API input format (JSONL).

Input:  data/01_processed_quality/quality_{split}.jsonl
Output: data/02_batch_inference_input/quality_{split}_batch.jsonl

Each output line has the form:
  {"custom_id": "<question_unique_id>", "body": {"messages": [...], "max_tokens": 512}}

Usage:
    python scripts/prepare_batch_input.py                    # process both train & test
    python scripts/prepare_batch_input.py --splits test      # process only test
    python scripts/prepare_batch_input.py --model gpt-4o     # specify model in body
"""

import argparse
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_ROOT / "data" / "01_processed_quality"
OUTPUT_DIR = PROJECT_ROOT / "data" / "02_batch_inference_input"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers multiple-choice reading comprehension questions.\n"
    "First, provide your reasoning step-by-step.\n"
    "Then, state the final answer on a new line in the exact format:\n"
    "Answer: (X)\n"
    "where X is one of A, B, C, or D."
)


def format_user_message(article: str, question: str, options: list[str]) -> str:
    options_formatted = "\n".join(
        f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options)
    )
    return (
        f"Read the following article and answer the question.\n\n"
        f"Article:\n{article}\n\n"
        f"Question: {question}\n"
        f"{options_formatted}"
    )


def build_batch_record(
    record: dict,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """Build a single Batch API request record."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": format_user_message(
                record["article"], record["question"], record["options"]
            ),
        },
    ]

    body: dict = {"messages": messages, "max_tokens": max_tokens}
    if model:
        body["model"] = model
    if temperature is not None:
        body["temperature"] = temperature

    return {
        "custom_id": record["question_unique_id"],
        "body": body,
    }


def process_split(split: str, model: str | None, max_tokens: int, temperature: float):
    input_path = INPUT_DIR / f"quality_{split}.jsonl"
    output_path = OUTPUT_DIR / f"quality_{split}_batch.jsonl"

    if not input_path.exists():
        print(f"[SKIP] Input file not found: {input_path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            record = json.loads(line)
            batch_record = build_batch_record(record, model, max_tokens, temperature)
            # Use ASCII-escaped JSON output so special Unicode separators
            # (e.g. U+2028/U+2029) are written as escape sequences, which avoids
            # VS Code unusual line terminator warnings when opening JSONL files.
            fout.write(json.dumps(batch_record) + "\n")
            count += 1

    print(f"[DONE] {split}: {count} records -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert processed QuALITY data to OpenAI Batch API input format."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Which splits to process (default: train test).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to include in the request body (e.g. gpt-4o). "
        "Omit to leave it unset (useful when the model is specified at batch-submit time).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens for generation (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy).",
    )
    args = parser.parse_args()

    for split in args.splits:
        process_split(split, args.model, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
