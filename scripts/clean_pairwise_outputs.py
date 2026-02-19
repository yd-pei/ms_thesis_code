"""
Clean pairwise response files by keeping only informative disagreement samples.

Filtering rules (all must hold):
1) Both candidate output labels are not null.
2) Exactly one candidate is correct w.r.t. gold_label
   (drop rows where both are correct or both are wrong).

Default:
- Input directory:  data/03_pairwise_output
- Output directory: data/04_clean_pairwise_output

Usage:
  python3 scripts/clean_pairwise_outputs.py

  python3 scripts/clean_pairwise_outputs.py \
      --input-file data/03_pairwise_output/llama31_8b__ds_v3.jsonl

  python3 scripts/clean_pairwise_outputs.py \
      --input-dir data/03_pairwise_output \
      --output-dir data/04_clean_pairwise_output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "03_pairwise_output"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "04_clean_pairwise_output"


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().upper()
        if not text:
            return None
        if text in {"NULL", "NONE", "NAN"}:
            return None
        return text
    return str(value).strip().upper() or None


def get_candidate_label_fields(record: dict[str, Any]) -> list[str]:
    return [key for key in record.keys() if key.endswith("_output_label")]


def should_keep_record(record: dict[str, Any], label_field_1: str, label_field_2: str) -> bool:
    gold = normalize_label(record.get("gold_label"))
    if gold is None:
        return False

    label_1 = normalize_label(record.get(label_field_1))
    label_2 = normalize_label(record.get(label_field_2))

    if label_1 is None or label_2 is None:
        return False

    is_correct_1 = label_1 == gold
    is_correct_2 = label_2 == gold

    return is_correct_1 ^ is_correct_2


def clean_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    total = 0
    kept = 0
    label_field_1: str | None = None
    label_field_2: str | None = None

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error in {input_path} line {lineno}: {exc}") from exc

            if label_field_1 is None or label_field_2 is None:
                label_fields = get_candidate_label_fields(record)
                if len(label_fields) != 2:
                    raise ValueError(
                        f"Expected exactly 2 '*_output_label' fields in {input_path} line {lineno}, "
                        f"got {len(label_fields)}: {label_fields}"
                    )
                label_field_1, label_field_2 = label_fields[0], label_fields[1]

            if should_keep_record(record, label_field_1, label_field_2):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    return total, kept


def iter_input_files(input_dir: Path) -> list[Path]:
    return sorted([path for path in input_dir.glob("*.jsonl") if path.is_file()])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter pairwise JSONL files to keep non-null labels with exactly one correct side."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing pairwise JSONL files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Single pairwise JSONL file to clean. If provided, --input-dir is ignored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for cleaned JSONL files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        input_files = [args.input_file]
    else:
        if not args.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        input_files = iter_input_files(args.input_dir)

    if not input_files:
        print("No input JSONL files found.")
        return

    total_all = 0
    kept_all = 0

    for input_path in input_files:
        output_path = args.output_dir / input_path.name
        total, kept = clean_file(input_path, output_path)
        total_all += total
        kept_all += kept
        print(f"{input_path.name}: kept {kept}/{total} -> {output_path}")

    print(f"Done. kept {kept_all}/{total_all} records across {len(input_files)} file(s).")


if __name__ == "__main__":
    main()
