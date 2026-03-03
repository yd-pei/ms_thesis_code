"""
Merge cleaned inference outputs into pairwise comparison format.

The primary model always comes first in the output filename and field names.
Only questions present in BOTH files (intersection) are included.

Output fields:
  - question_unique_id
  - gold_label
  - {primary}_output_label
  - {primary}_output_reason
  - {opponent}_output_label
  - {opponent}_output_reason

Default:
- Input directory:  data/025_cleaned_inference
- Output directory: data/035_pairwise_output

Usage:
  # Merge one pair
  uv run python scripts/answer_extraction/merge_pairwise.py \
      --primary data/025_cleaned_inference/llama31_8b.jsonl \
      --opponent data/025_cleaned_inference/ds_v3.jsonl

  # Merge primary against ALL other files in a directory
  uv run python scripts/answer_extraction/merge_pairwise.py \
      --primary data/025_cleaned_inference/llama31_8b.jsonl \
      --opponent-dir data/025_cleaned_inference

  # Custom model names
  uv run python scripts/answer_extraction/merge_pairwise.py \
      --primary data/025_cleaned_inference/llama31_8b.jsonl \
      --opponent data/025_cleaned_inference/ds_v3.jsonl \
      --primary-name llama31_8b --opponent-name deepseek_v3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "025_cleaned_inference"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "035_pairwise_output"


def sanitize_model_name(name: str) -> str:
    """Convert model/file name into a safe field prefix."""
    stem = Path(name).stem
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
    return safe or "candidate"


def load_jsonl_by_qid(file_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with file_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error in {file_path} line {lineno}: {exc}"
                ) from exc
            qid = row.get("question_unique_id")
            if not qid:
                continue
            records[qid] = row
    return records


def merge_pairwise(
    primary_records: dict[str, dict[str, Any]],
    opponent_records: dict[str, dict[str, Any]],
    primary_name: str,
    opponent_name: str,
) -> list[dict[str, Any]]:
    common_qids = sorted(set(primary_records) & set(opponent_records))
    merged: list[dict[str, Any]] = []

    for qid in common_qids:
        rec_p = primary_records[qid]
        rec_o = opponent_records[qid]

        gold_p = rec_p.get("gold_label")
        gold_o = rec_o.get("gold_label")
        gold_label = gold_p if gold_p is not None else gold_o

        merged.append(
            {
                "question_unique_id": qid,
                "gold_label": gold_label,
                f"{primary_name}_output_label": rec_p.get("output_label"),
                f"{primary_name}_output_reason": rec_p.get("reason"),
                f"{opponent_name}_output_label": rec_o.get("output_label"),
                f"{opponent_name}_output_reason": rec_o.get("reason"),
            }
        )

    return merged


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def do_merge(
    primary_path: Path,
    opponent_path: Path,
    primary_name: str,
    opponent_name: str,
    output_dir: Path,
) -> None:
    records_p = load_jsonl_by_qid(primary_path)
    records_o = load_jsonl_by_qid(opponent_path)

    merged = merge_pairwise(records_p, records_o, primary_name, opponent_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{primary_name}__{opponent_name}.jsonl"
    output_path = output_dir / output_name

    write_jsonl(merged, output_path)

    print(
        f"{primary_name} ({len(records_p)}) x {opponent_name} ({len(records_o)}) "
        f"-> {len(merged)} pairs  =>  {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge cleaned inference files into pairwise format. "
        "Primary model is always first in filenames and fields."
    )
    parser.add_argument(
        "--primary",
        required=True,
        type=Path,
        help="Path to the primary model JSONL file.",
    )

    # Opponent: either a single file or a whole directory
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--opponent",
        type=Path,
        help="Path to a single opponent JSONL file.",
    )
    group.add_argument(
        "--opponent-dir",
        type=Path,
        help="Directory of opponent JSONL files. Merges primary against every other file in the directory.",
    )

    parser.add_argument(
        "--primary-name",
        type=str,
        default=None,
        help="Field prefix for primary model. Default: inferred from filename.",
    )
    parser.add_argument(
        "--opponent-name",
        type=str,
        default=None,
        help="Field prefix for opponent model (only for single --opponent). Default: inferred from filename.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    if not args.primary.exists():
        raise FileNotFoundError(f"Primary file not found: {args.primary}")

    p_name = sanitize_model_name(args.primary_name or args.primary.stem)

    if args.opponent:
        # Single opponent mode
        if not args.opponent.exists():
            raise FileNotFoundError(f"Opponent file not found: {args.opponent}")
        o_name = sanitize_model_name(args.opponent_name or args.opponent.stem)
        do_merge(args.primary, args.opponent, p_name, o_name, args.output_dir)

    else:
        # Batch mode: primary vs every other file in opponent-dir
        if not args.opponent_dir.exists():
            raise FileNotFoundError(f"Opponent directory not found: {args.opponent_dir}")

        opponent_files = sorted(
            p for p in args.opponent_dir.glob("*.jsonl")
            if p.is_file() and p.resolve() != args.primary.resolve()
        )

        if not opponent_files:
            print("No opponent JSONL files found.")
            return

        print(f"Batch mode: {p_name} vs {len(opponent_files)} opponent(s)\n")
        for opp_path in opponent_files:
            o_name = sanitize_model_name(opp_path.stem)
            if o_name == p_name:
                continue
            do_merge(args.primary, opp_path, p_name, o_name, args.output_dir)

        print(f"\nDone. {len(opponent_files)} pair(s) written to {args.output_dir}")


if __name__ == "__main__":
    main()
