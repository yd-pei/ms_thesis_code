"""
Merge two model response JSONL files from data/02_quality_response by question_unique_id.

Output fields:
  - question_unique_id
  - gold_label
  - {candidate_1_model}_output_label
  - {candidate_1_model}_output_reason
  - {candidate_2_model}_output_label
  - {candidate_2_model}_output_reason

Default output directory:
  data/03_pairwise_output

Usage:
  uv run python scripts/merge_pairwise_responses.py \
      --candidate-1 data/02_quality_response/llama31_8b.jsonl \
      --candidate-2 data/02_quality_response/ds_v3.jsonl

Optional custom model field names:
  uv run python scripts/merge_pairwise_responses.py \
      --candidate-1 data/02_quality_response/model_a.jsonl \
      --candidate-2 data/02_quality_response/model_b.jsonl \
      --candidate-1-name llama31_8b \
      --candidate-2-name deepseek_v3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "03_pairwise_output"


def sanitize_model_name(name: str) -> str:
    """Convert model/file name into a safe field prefix."""
    stem = Path(name).stem
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
    return safe or "candidate"


def load_jsonl_by_qid(file_path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with file_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error in {file_path} line {lineno}: {exc}") from exc

            qid = row.get("question_unique_id")
            if not qid:
                continue

            records[qid] = row
    return records


def merge_pairwise(
    candidate_1_records: Dict[str, Dict[str, Any]],
    candidate_2_records: Dict[str, Dict[str, Any]],
    candidate_1_name: str,
    candidate_2_name: str,
) -> list[Dict[str, Any]]:
    common_qids = sorted(set(candidate_1_records).intersection(candidate_2_records))
    merged: list[Dict[str, Any]] = []

    for qid in common_qids:
        rec1 = candidate_1_records[qid]
        rec2 = candidate_2_records[qid]

        gold_1 = rec1.get("gold_label")
        gold_2 = rec2.get("gold_label")
        gold_label = gold_1 if gold_1 is not None else gold_2

        merged.append(
            {
                "question_unique_id": qid,
                "gold_label": gold_label,
                f"{candidate_1_name}_output_label": rec1.get("output_label"),
                f"{candidate_1_name}_output_reason": rec1.get("reason"),
                f"{candidate_2_name}_output_label": rec2.get("output_label"),
                f"{candidate_2_name}_output_reason": rec2.get("reason"),
            }
        )

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two response JSONL files by question_unique_id into pairwise format."
    )
    parser.add_argument(
        "--candidate-1",
        required=True,
        type=Path,
        help="Path to candidate 1 JSONL file (e.g., data/02_quality_response/llama31_8b.jsonl)",
    )
    parser.add_argument(
        "--candidate-2",
        required=True,
        type=Path,
        help="Path to candidate 2 JSONL file (e.g., data/02_quality_response/ds_v3.jsonl)",
    )
    parser.add_argument(
        "--candidate-1-name",
        type=str,
        default=None,
        help="Optional field prefix for candidate 1. Default: inferred from filename stem.",
    )
    parser.add_argument(
        "--candidate-2-name",
        type=str,
        default=None,
        help="Optional field prefix for candidate 2. Default: inferred from filename stem.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional output filename. Default: <candidate1>__<candidate2>.jsonl",
    )

    args = parser.parse_args()

    if not args.candidate_1.exists():
        raise FileNotFoundError(f"Candidate 1 file not found: {args.candidate_1}")
    if not args.candidate_2.exists():
        raise FileNotFoundError(f"Candidate 2 file not found: {args.candidate_2}")

    candidate_1_name = sanitize_model_name(args.candidate_1_name or args.candidate_1.stem)
    candidate_2_name = sanitize_model_name(args.candidate_2_name or args.candidate_2.stem)

    records_1 = load_jsonl_by_qid(args.candidate_1)
    records_2 = load_jsonl_by_qid(args.candidate_2)

    merged = merge_pairwise(records_1, records_2, candidate_1_name, candidate_2_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_file or f"{candidate_1_name}__{candidate_2_name}.jsonl"
    output_path = args.output_dir / output_name

    with output_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"candidate_1 records: {len(records_1)}")
    print(f"candidate_2 records: {len(records_2)}")
    print(f"merged (common question_unique_id): {len(merged)}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
