"""
Analyze distribution of judge_output values in pairwise JSONL data.

For each file (and overall), report counts and rates of:
- judge_output == "1"
- judge_output == "2"
- missing judge_output
- invalid judge_output (not in {"1", "2"})

Usage:
  python3 scripts/judge_analysis/analyze_judge_output_distribution.py \
      --input-file data/12_output_restyled/ds/llama31_8b.jsonl

  python3 scripts/judge_analysis/analyze_judge_output_distribution.py \
      --input-dir data/12_output_restyled/ds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "12_output_restyled" / "ds"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def normalize_judge_output(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"1", "2"}:
        return text
    return "INVALID"


def collect_files(input_dir: Path, input_file: Path | None) -> list[Path]:
    if input_file is not None:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        return [input_file]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    return sorted([p for p in input_dir.glob("*.jsonl") if p.is_file()])


def analyze_file(path: Path) -> dict[str, Any]:
    total_records = 0
    count_1 = 0
    count_2 = 0
    missing = 0
    invalid = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            total_records += 1

            judge = normalize_judge_output(row.get("judge_output"))
            if judge is None:
                missing += 1
            elif judge == "INVALID":
                invalid += 1
            elif judge == "1":
                count_1 += 1
            else:
                count_2 += 1

    valid_total = count_1 + count_2

    return {
        "file": str(path),
        "file_name": path.name,
        "total_records": total_records,
        "valid_total": valid_total,
        "count_1": count_1,
        "count_2": count_2,
        "missing": missing,
        "invalid": invalid,
        "rate_1_in_valid": (count_1 / valid_total) if valid_total else None,
        "rate_2_in_valid": (count_2 / valid_total) if valid_total else None,
        "rate_1_in_total": (count_1 / total_records) if total_records else None,
        "rate_2_in_total": (count_2 / total_records) if total_records else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze judge_output (1/2) distribution for pairwise JSONL data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional single JSONL file to analyze.",
    )
    args = parser.parse_args()

    files = collect_files(args.input_dir, args.input_file)
    if not files:
        print("No JSONL files found.")
        return

    all_results = [analyze_file(path) for path in files]

    sum_total = sum(r["total_records"] for r in all_results)
    sum_valid = sum(r["valid_total"] for r in all_results)
    sum_1 = sum(r["count_1"] for r in all_results)
    sum_2 = sum(r["count_2"] for r in all_results)
    sum_missing = sum(r["missing"] for r in all_results)
    sum_invalid = sum(r["invalid"] for r in all_results)

    print("=== Per-file judge_output distribution ===")
    for r in all_results:
        print(f"\n[{r['file_name']}]")
        print(f"  total_records={r['total_records']}")
        print(f"  valid_total={r['valid_total']}")
        print(
            f"  judge_output=1: {r['count_1']} "
            f"(in_valid={fmt_pct(r['rate_1_in_valid'])}, in_total={fmt_pct(r['rate_1_in_total'])})"
        )
        print(
            f"  judge_output=2: {r['count_2']} "
            f"(in_valid={fmt_pct(r['rate_2_in_valid'])}, in_total={fmt_pct(r['rate_2_in_total'])})"
        )
        print(f"  missing={r['missing']}")
        print(f"  invalid={r['invalid']}")

    overall_rate_1_valid = (sum_1 / sum_valid) if sum_valid else None
    overall_rate_2_valid = (sum_2 / sum_valid) if sum_valid else None
    overall_rate_1_total = (sum_1 / sum_total) if sum_total else None
    overall_rate_2_total = (sum_2 / sum_total) if sum_total else None

    print("\n=== Overall ===")
    print(f"files={len(all_results)}")
    print(f"total_records={sum_total}")
    print(f"valid_total={sum_valid}")
    print(
        f"judge_output=1: {sum_1} "
        f"(in_valid={fmt_pct(overall_rate_1_valid)}, in_total={fmt_pct(overall_rate_1_total)})"
    )
    print(
        f"judge_output=2: {sum_2} "
        f"(in_valid={fmt_pct(overall_rate_2_valid)}, in_total={fmt_pct(overall_rate_2_total)})"
    )
    print(f"missing={sum_missing}")
    print(f"invalid={sum_invalid}")


if __name__ == "__main__":
    main()
