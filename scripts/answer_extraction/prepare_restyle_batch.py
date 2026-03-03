"""
Extract prompts from cleaned inference outputs (025) and build Gemini Batch
API JSONL input for synonym restyle.

Combines the logic of 07 (extract_prompt_fields) and 08 (prepare_gemini_batch)
into a single step for the 025 pipeline.

For each record in 025, looks up the question text from 01_processed_quality,
then builds a Gemini batch request asking for synonym replacements.

Default:
- Input directory:  data/025_cleaned_inference
- Question source:  data/01_processed_quality/quality_train.jsonl
- Output directory:  data/075_restyle_batch_input

Usage:
  uv run python scripts/answer_extraction/prepare_restyle_batch.py

  uv run python scripts/answer_extraction/prepare_restyle_batch.py \
      --input-file data/025_cleaned_inference/llama33_70b.jsonl

  uv run python scripts/answer_extraction/prepare_restyle_batch.py \
      --input-dir data/025_cleaned_inference \
      --output-dir data/075_restyle_batch_input
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "025_cleaned_inference"
DEFAULT_QUALITY_TRAIN = PROJECT_ROOT / "data" / "01_processed_quality" / "quality_train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "075_restyle_batch_input"

SYSTEM_PROMPT = (
    "You are a helpful assistant that helps rewrites sentences. "
    "Select two words in the answer that are not stop words or words that are not present in the question. "
    "Then, suggest their replacements with their synonyms in the answer sentence - "
    "make sure the suggested words do not change the meaning of the answer."
)

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "selected_words": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "replacements": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["selected_words", "replacements"],
}


def load_question_map(quality_train_path: Path) -> dict[str, str]:
    """Load question_unique_id -> question text mapping."""
    if not quality_train_path.exists():
        raise FileNotFoundError(f"quality_train file not found: {quality_train_path}")

    question_map: dict[str, str] = {}
    with quality_train_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_unique_id")
            question = row.get("question")
            if qid and question:
                question_map[str(qid)] = str(question)

    return question_map


def build_request_record(question: str, answer: str, qid: str) -> dict[str, Any]:
    """Build a single Gemini batch API request object."""
    user_text = f"Question:\n{question}\n\nAnswer:\n{answer}"

    return {
        "key": qid,
        "request": {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_text}],
                }
            ],
            "generation_config": {
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "response_schema": RESPONSE_SCHEMA,
            },
        },
    }


def process_file(
    input_file: Path,
    output_file: Path,
    question_map: dict[str, str],
) -> dict[str, int]:
    stats = {
        "total": 0,
        "written": 0,
        "missing_question": 0,
        "missing_reason": 0,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open("r", encoding="utf-8") as fin, \
         output_file.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1
            row = json.loads(line)

            qid = str(row.get("question_unique_id", ""))
            if not qid:
                continue

            question = question_map.get(qid)
            if not question:
                stats["missing_question"] += 1
                continue

            reason = row.get("reason")
            if not reason:
                stats["missing_reason"] += 1
                continue

            record = build_request_record(
                question=question,
                answer=str(reason),
                qid=qid,
            )
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Gemini batch restyle input from cleaned inference outputs (025)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory of cleaned inference JSONL files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Single JSONL file. If provided, --input-dir is ignored.",
    )
    parser.add_argument(
        "--quality-train",
        type=Path,
        default=DEFAULT_QUALITY_TRAIN,
        help=f"Path to quality_train.jsonl (default: {DEFAULT_QUALITY_TRAIN})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    print("Loading question map...")
    question_map = load_question_map(args.quality_train)
    print(f"Loaded {len(question_map)} questions.\n")

    if args.input_file:
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        input_files = [args.input_file]
    else:
        if not args.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        input_files = sorted(p for p in args.input_dir.glob("*.jsonl") if p.is_file())

    if not input_files:
        print("No input JSONL files found.")
        return

    total_written = 0
    total_rows = 0

    for input_file in input_files:
        output_file = args.output_dir / input_file.name
        stats = process_file(input_file, output_file, question_map)

        total_rows += stats["total"]
        total_written += stats["written"]

        print(
            f"{input_file.name}: {stats['written']}/{stats['total']} written"
            + (f", {stats['missing_question']} missing question" if stats["missing_question"] else "")
            + (f", {stats['missing_reason']} missing reason" if stats["missing_reason"] else "")
        )

    print(f"\nDone. {total_written}/{total_rows} records written across {len(input_files)} file(s).")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
