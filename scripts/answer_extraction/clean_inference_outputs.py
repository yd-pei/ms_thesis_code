"""
Clean inference output files by re-extracting answer labels and filtering out
records where the label cannot be extracted.

For records with a `raw_generation` field, the answer label is re-extracted
using the latest `extract_answer_components()` (last-match, markdown-aware).
For records without `raw_generation`, the existing `output_label` is kept as-is.

Records whose `output_label` is None / null / empty after extraction are dropped.

Default:
- Input directory:  data/02_quality_response/raw
- Output directory: data/025_cleaned_inference

Usage:
  uv run python scripts/answer_extraction/clean_inference_outputs.py

  uv run python scripts/answer_extraction/clean_inference_outputs.py \
      --input-file data/02_quality_response/raw/llama31_8b.jsonl

  uv run python scripts/answer_extraction/clean_inference_outputs.py \
      --input-dir data/02_quality_response/raw \
      --output-dir data/025_cleaned_inference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # scripts/data_preprocess -> scripts -> project root
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "02_quality_response" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "025_cleaned_inference"

# Add project src to path so we can import from align_lab
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from align_lab.inference.engine import extract_answer_components  # noqa: E402


def normalize_label(value: Any) -> str | None:
    """Return a cleaned uppercase label, or None if empty / null."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in {"NULL", "NONE", "NAN"}:
        return None
    return text


def clean_file(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    """
    Clean a single JSONL file.

    Returns:
        (total, kept, re_extracted) counts.
    """
    total = 0
    kept = 0
    re_extracted = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error in {input_path} line {lineno}: {exc}"
                ) from exc

            # Re-extract from raw_generation if available
            raw_gen = record.get("raw_generation")
            if raw_gen is not None:
                new_label, new_reason = extract_answer_components(raw_gen)
                record["output_label"] = new_label
                record["reason"] = new_reason
                re_extracted += 1

            # Filter: keep only records with a valid output_label
            label = normalize_label(record.get("output_label"))
            if label is None:
                continue

            record["output_label"] = label
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    return total, kept, re_extracted


def iter_jsonl_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.jsonl") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-extract answer labels from inference outputs and filter invalid records."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing inference JSONL files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Single JSONL file to clean. If provided, --input-dir is ignored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for cleaned files (default: {DEFAULT_OUTPUT_DIR})",
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
        input_files = iter_jsonl_files(args.input_dir)

    if not input_files:
        print("No input JSONL files found.")
        return

    total_all = 0
    kept_all = 0

    for input_path in input_files:
        output_path = args.output_dir / input_path.name
        total, kept, re_extracted = clean_file(input_path, output_path)
        total_all += total
        kept_all += kept
        dropped = total - kept
        print(
            f"{input_path.name}: {kept}/{total} kept, {dropped} dropped"
            + (f", {re_extracted} re-extracted from raw_generation" if re_extracted else "")
        )

    print(f"\nDone. {kept_all}/{total_all} records kept across {len(input_files)} file(s).")


if __name__ == "__main__":
    main()
