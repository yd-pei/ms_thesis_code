"""
Generate Gemini Batch API JSONL input from extracted prompt files.

Input files (default):
  data/07_extracted_prompt/*/with_ds/*.jsonl

Each input row should contain:
  - question_unique_id
  - model_output_reason

Question text is looked up from:
  data/01_processed_quality/quality_train.jsonl

Output files (default):
  data/08_gemini_api/with_ds/<same_filename>.jsonl

Each output line is a Gemini Batch request object, e.g.:
{
  "key": "<question_unique_id>",
  "request": {
    "system_instruction": {"parts": [{"text": "..."}]},
    "contents": [{"role": "user", "parts": [{"text": "Question:\n...\n\nAnswer:\n..."}]}],
    "generation_config": {
      "temperature": 0.0,
      "response_mime_type": "application/json",
      "response_schema": { ... }
    }
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_EXTRACTED_ROOT = PROJECT_ROOT / "data" / "07_extracted_prompt" / "raw"
DEFAULT_QUALITY_TRAIN = PROJECT_ROOT / "data" / "01_processed_quality" / "quality_train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "08_gemini_api" / "raw"

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
    if not quality_train_path.exists():
        raise FileNotFoundError(f"quality_train file not found: {quality_train_path}")

    question_map: dict[str, str] = {}
    with quality_train_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error in {quality_train_path} line {lineno}: {exc}"
                ) from exc

            qid = row.get("question_unique_id")
            question = row.get("question")
            if not qid or not question:
                continue

            question_map[str(qid)] = str(question)

    return question_map


def collect_input_files(extracted_root: Path) -> list[Path]:
    if not extracted_root.exists():
        raise FileNotFoundError(f"Extracted prompt root not found: {extracted_root}")

    files = sorted(extracted_root.glob("*/with_ds/*.jsonl"))
    return [p for p in files if p.is_file()]


def build_request_record(question: str, answer: str, qid: str) -> dict[str, Any]:
    user_text = f"Question:\n{question}\n\nAnswer:\n{answer}"

    return {
        "key": qid,
        "request": {
            "system_instruction": {
                "parts": [
                    {
                        "text": SYSTEM_PROMPT,
                    }
                ]
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
        "missing_qid": 0,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open("r", encoding="utf-8") as fin, output_file.open(
        "w", encoding="utf-8"
    ) as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error in {input_file} line {lineno}: {exc}"
                ) from exc

            qid = row.get("question_unique_id")
            if not qid:
                stats["missing_qid"] += 1
                continue

            qid = str(qid)
            question = question_map.get(qid)
            if not question:
                stats["missing_question"] += 1
                continue

            answer_reason = row.get("model_output_reason")
            if not answer_reason:
                stats["missing_reason"] += 1
                continue

            request_record = build_request_record(
                question=question,
                answer=str(answer_reason),
                qid=qid,
            )
            fout.write(json.dumps(request_record, ensure_ascii=False) + "\n")
            stats["written"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Gemini Batch API JSONL input from extracted prompt files."
    )
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=DEFAULT_EXTRACTED_ROOT,
        help=f"Root directory of extracted prompts (default: {DEFAULT_EXTRACTED_ROOT})",
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

    question_map = load_question_map(args.quality_train)
    input_files = collect_input_files(args.extracted_root)

    if not input_files:
        print("No input files found under extracted root.")
        return

    total_written = 0
    total_rows = 0

    for input_file in input_files:
        output_file = args.output_dir / input_file.name
        stats = process_file(input_file, output_file, question_map)

        total_rows += stats["total"]
        total_written += stats["written"]

        print(
            f"[done] {input_file.name}: total={stats['total']} written={stats['written']} "
            f"missing_qid={stats['missing_qid']} missing_question={stats['missing_question']} "
            f"missing_reason={stats['missing_reason']} -> {output_file}"
        )

    print("\n=== Summary ===")
    print(f"files_processed={len(input_files)}")
    print(f"total_rows={total_rows}")
    print(f"total_written={total_written}")


if __name__ == "__main__":
    main()
