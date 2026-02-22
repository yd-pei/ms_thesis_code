"""
Check whether judge_output values are valid (only '1' or '2') in pairwise response JSONL files.

Default input directory:
    data/05_pairwise_response

Usage:
    python3 scripts/judge_analysis/check_judge_output_values.py
    python3 scripts/judge_analysis/check_judge_output_values.py --input-file data/05_pairwise_response/llama31_70_reverse.jsonl
    python3 scripts/judge_analysis/check_judge_output_values.py --max-examples 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "05_pairwise_response"
VALID_VALUES = {"1", "2"}


def normalize_judge_output(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text


def analyze_file(file_path: Path, max_examples: int) -> dict[str, Any]:
    total = 0
    valid = 0
    invalid = 0
    missing = 0
    examples: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "line": lineno,
                            "question_unique_id": None,
                            "judge_output": "<JSON_ERROR>",
                            "reason": f"JSON decode error: {exc}",
                        }
                    )
                continue

            total += 1
            judge_output = normalize_judge_output(row.get("judge_output"))

            if judge_output is None:
                missing += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "line": lineno,
                            "question_unique_id": row.get("question_unique_id"),
                            "judge_output": None,
                            "reason": "missing_or_empty",
                        }
                    )
                continue

            if judge_output in VALID_VALUES:
                valid += 1
            else:
                invalid += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "line": lineno,
                            "question_unique_id": row.get("question_unique_id"),
                            "judge_output": judge_output,
                            "reason": "not_in_{1,2}",
                        }
                    )

    return {
        "file": str(file_path),
        "total_records": total,
        "valid_records": valid,
        "invalid_records": invalid,
        "missing_records": missing,
        "is_all_valid": (invalid == 0 and missing == 0),
        "examples": examples,
    }


def collect_input_files(input_dir: Path, input_file: Path | None) -> list[Path]:
    if input_file is not None:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        return [input_file]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    return sorted([path for path in input_dir.glob("*.jsonl") if path.is_file()])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether judge_output values are only '1' or '2'."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing JSONL files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional single JSONL file to check.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Max number of invalid/missing examples to print per file.",
    )
    args = parser.parse_args()

    files = collect_input_files(args.input_dir, args.input_file)
    if not files:
        print("No JSONL files found.")
        return

    total_files = 0
    all_valid_files = 0
    total_records = 0
    total_invalid = 0
    total_missing = 0

    for path in files:
        total_files += 1
        result = analyze_file(path, max_examples=args.max_examples)

        total_records += result["total_records"]
        total_invalid += result["invalid_records"]
        total_missing += result["missing_records"]

        status = "PASS" if result["is_all_valid"] else "FAIL"
        if result["is_all_valid"]:
            all_valid_files += 1

        print(f"\n[{status}] {path.name}")
        print(
            f"  total={result['total_records']} valid={result['valid_records']} "
            f"invalid={result['invalid_records']} missing={result['missing_records']}"
        )

        if result["examples"]:
            print("  examples:")
            for ex in result["examples"]:
                print(
                    f"    line={ex['line']} qid={ex['question_unique_id']} "
                    f"judge_output={ex['judge_output']!r} reason={ex['reason']}"
                )

    print("\n=== Summary ===")
    print(f"files_checked={total_files}")
    print(f"files_all_valid={all_valid_files}")
    print(f"total_records={total_records}")
    print(f"total_invalid={total_invalid}")
    print(f"total_missing={total_missing}")


if __name__ == "__main__":
    main()
