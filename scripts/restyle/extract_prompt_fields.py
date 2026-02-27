"""
Extract model-under-test prompt fields from position-independent judge outputs.

From each input JSONL row, keep only:
- question_unique_id
- gold_label
- model_label
- model_output_reason

The model-under-test is inferred from the input filename:
  <model_name>_judge.jsonl

Example:
  input file:  llama31_8b_judge.jsonl
  source keys: llama31_8b_output_label / llama31_8b_output_reason

Default:
- Input directory:  data/06_position_independent_response/greedy
- Output root:     data/07_extracted_prompt
- Output path per file:
    data/07_extracted_prompt/<model_name>/with_ds/<model_name>_judge.jsonl

Usage:
  python3 scripts/judge_analysis/extract_prompt_fields.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "06_position_independent_response" / "greedy"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "07_extracted_prompt"


def infer_model_name(file_path: Path) -> str:
    stem = file_path.stem
    suffix = "_judge"
    if not stem.endswith(suffix):
        raise ValueError(
            f"Input filename must end with '{suffix}.jsonl': {file_path.name}"
        )
    return stem[: -len(suffix)]


def extract_row(row: dict[str, Any], model_name: str) -> dict[str, Any]:
    label_key = f"{model_name}_output_label"
    reason_key = f"{model_name}_output_reason"

    return {
        "question_unique_id": row.get("question_unique_id"),
        "gold_label": row.get("gold_label"),
        "model_label": row.get(label_key),
        "model_output_reason": row.get(reason_key),
    }


def process_file(input_file: Path, output_root: Path) -> tuple[Path, int]:
    model_name = infer_model_name(input_file)
    output_dir = output_root / model_name / "with_ds"
    output_file = output_dir / input_file.name

    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_file.open("r", encoding="utf-8") as fin, output_file.open(
        "w", encoding="utf-8"
    ) as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error in {input_file} line {lineno}: {exc}"
                ) from exc

            extracted = extract_row(row, model_name)
            fout.write(json.dumps(extracted, ensure_ascii=False) + "\n")
            count += 1

    return output_file, count


def collect_input_files(input_dir: Path) -> list[Path]:
    files = sorted([p for p in input_dir.glob("*_judge.jsonl") if p.is_file()])
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract qid/gold/model_label/model_output_reason from position-independent JSONL files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    input_files = collect_input_files(args.input_dir)
    if not input_files:
        print("No input files matched '*_judge.jsonl'.")
        return

    total_rows = 0
    for input_file in input_files:
        output_file, count = process_file(input_file, args.output_root)
        total_rows += count
        print(f"[done] {input_file.name}: rows={count} -> {output_file}")

    print("\n=== Summary ===")
    print(f"files_processed={len(input_files)}")
    print(f"total_rows_written={total_rows}")


if __name__ == "__main__":
    main()
