"""
Evaluate per-model accuracy within cleaned pairwise files.

For each pairwise JSONL file (with fields {model}_output_label and gold_label),
report each model's accuracy — i.e. how often each side is the correct one
among the kept (exactly-one-correct) pairs.

Default:
- Input directory: data/045_clean_pairwise_output

Usage:
  uv run python scripts/answer_extraction/eval_pairwise_accuracy.py

  uv run python scripts/answer_extraction/eval_pairwise_accuracy.py \
      --input-dir data/045_clean_pairwise_output

  uv run python scripts/answer_extraction/eval_pairwise_accuracy.py \
      --input-file data/045_clean_pairwise_output/llama33_70b__llama31_8b.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "045_clean_pairwise_output"


def get_label_fields(record: dict[str, Any]) -> tuple[str, str]:
    """Return the two *_output_label field names."""
    fields = [k for k in record if k.endswith("_output_label")]
    if len(fields) != 2:
        raise ValueError(f"Expected 2 *_output_label fields, got {len(fields)}: {fields}")
    return fields[0], fields[1]


def model_name_from_field(field: str) -> str:
    """'llama33_70b_output_label' -> 'llama33_70b'"""
    return field.removesuffix("_output_label")


def evaluate_file(path: Path) -> dict[str, Any]:
    total = 0
    field_1: str | None = None
    field_2: str | None = None
    correct_1 = 0
    correct_2 = 0

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            if field_1 is None:
                field_1, field_2 = get_label_fields(record)

            total += 1
            gold = record.get("gold_label", "").strip().upper()
            label_1 = (record.get(field_1) or "").strip().upper()
            label_2 = (record.get(field_2) or "").strip().upper()

            if label_1 == gold:
                correct_1 += 1
            if label_2 == gold:
                correct_2 += 1

    name_1 = model_name_from_field(field_1) if field_1 else "?"
    name_2 = model_name_from_field(field_2) if field_2 else "?"

    return {
        "file": path.name,
        "total": total,
        "primary": name_1,
        "opponent": name_2,
        "primary_correct": correct_1,
        "opponent_correct": correct_2,
    }


def fmt_pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "N/A"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate per-model accuracy in cleaned pairwise files."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Directory of pairwise JSONL files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--input-file", type=Path, default=None,
        help="Single pairwise JSONL file. If provided, --input-dir is ignored.",
    )
    args = parser.parse_args()

    if args.input_file:
        files = [args.input_file]
    else:
        files = sorted(args.input_dir.glob("*.jsonl"))

    if not files:
        print("No JSONL files found.")
        return

    results: list[dict[str, Any]] = []
    for f in files:
        results.append(evaluate_file(f))

    # Print per-file results
    for r in results:
        t = r["total"]
        print(f"\n{'=' * 60}")
        print(f"  {r['file']}  ({t} pairs)")
        print(f"{'=' * 60}")
        print(f"  {r['primary']:<20} {r['primary_correct']:>5}/{t}  {fmt_pct(r['primary_correct'], t):>7}")
        print(f"  {r['opponent']:<20} {r['opponent_correct']:>5}/{t}  {fmt_pct(r['opponent_correct'], t):>7}")

    # Summary table
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("  Summary")
        print(f"{'=' * 60}")
        print(f"  {'Pair':<35} {'Primary':>12} {'Opponent':>12} {'Pairs':>6}")
        print(f"  {'-' * 67}")
        for r in results:
            t = r["total"]
            p_str = f"{fmt_pct(r['primary_correct'], t)}"
            o_str = f"{fmt_pct(r['opponent_correct'], t)}"
            label = f"{r['primary']} vs {r['opponent']}"
            print(f"  {label:<35} {p_str:>12} {o_str:>12} {t:>6}")
        print()


if __name__ == "__main__":
    main()
