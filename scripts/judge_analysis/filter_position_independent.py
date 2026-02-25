"""
Filter position-independent judge responses by comparing normal vs reverse files.

Rule (keep):
- Same question_unique_id exists in both files.
- normal judge_output and reverse judge_output are both in {'1', '2'}.
- They are opposite after swapping answer positions:
    normal='1' <-> reverse='2', or normal='2' <-> reverse='1'.

Default:
- Input directory:  data/05_pairwise_response
- Output directory: data/06_position_independent_response

Usage:
  python3 scripts/judge_analysis/filter_position_independent.py

  python3 scripts/judge_analysis/filter_position_independent.py \
      --normal-file data/05_pairwise_response/llama31_70.jsonl \
      --reverse-file data/05_pairwise_response/llama31_70_reverse.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "05_pairwise_response" / "greedy"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "06_position_independent_response" / "greedy"
)
VALID = {"1", "2"}


def normalize_judge_output(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def load_jsonl_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    duplicates = 0

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error in {path} line {lineno}: {exc}"
                ) from exc

            qid = row.get("question_unique_id")
            if not qid:
                continue

            if qid in data:
                duplicates += 1
            data[qid] = row

    if duplicates:
        print(
            f"[warn] {path.name}: duplicate question_unique_id count={duplicates} (last occurrence kept)"
        )

    return data


def keep_position_independent(
    normal_map: dict[str, dict[str, Any]],
    reverse_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []

    stats = {
        "normal_total": len(normal_map),
        "reverse_total": len(reverse_map),
        "missing_in_reverse": 0,
        "invalid_judge_output": 0,
        "position_dependent": 0,
        "kept": 0,
    }

    for qid, normal_row in normal_map.items():
        reverse_row = reverse_map.get(qid)
        if reverse_row is None:
            stats["missing_in_reverse"] += 1
            continue

        normal_judge = normalize_judge_output(normal_row.get("judge_output"))
        reverse_judge = normalize_judge_output(reverse_row.get("judge_output"))

        if normal_judge not in VALID or reverse_judge not in VALID:
            stats["invalid_judge_output"] += 1
            continue

        if normal_judge == reverse_judge:
            stats["position_dependent"] += 1
            continue

        # valid and opposite -> position independent
        kept.append(normal_row)

    stats["kept"] = len(kept)
    return kept, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_pair(
    normal_file: Path, reverse_file: Path, output_file: Path
) -> dict[str, int]:
    normal_map = load_jsonl_by_qid(normal_file)
    reverse_map = load_jsonl_by_qid(reverse_file)

    kept_rows, stats = keep_position_independent(normal_map, reverse_map)
    write_jsonl(output_file, kept_rows)

    print(f"\n[done] {normal_file.name} vs {reverse_file.name}")
    print(
        "  "
        f"normal_total={stats['normal_total']} reverse_total={stats['reverse_total']} "
        f"kept={stats['kept']} missing_in_reverse={stats['missing_in_reverse']} "
        f"invalid_judge_output={stats['invalid_judge_output']} "
        f"position_dependent={stats['position_dependent']}"
    )
    print(f"  output={output_file}")

    return stats


def collect_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    normal_files = sorted(
        [
            p
            for p in input_dir.glob("*.jsonl")
            if p.is_file() and not p.stem.endswith("_reverse")
        ]
    )

    pairs: list[tuple[Path, Path]] = []
    for normal in normal_files:
        reverse = normal.with_name(f"{normal.stem}_reverse.jsonl")
        if reverse.exists():
            pairs.append((normal, reverse))
        else:
            print(f"[warn] reverse file not found for {normal.name}, skipped")

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep position-independent samples by comparing normal and reverse judge outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--normal-file", type=Path, default=None, help="Optional normal JSONL file"
    )
    parser.add_argument(
        "--reverse-file", type=Path, default=None, help="Optional reverse JSONL file"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output file (only valid when --normal-file and --reverse-file are set)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.normal_file or args.reverse_file:
        if not (args.normal_file and args.reverse_file):
            raise ValueError(
                "Please provide both --normal-file and --reverse-file together."
            )
        if not args.normal_file.exists():
            raise FileNotFoundError(f"Normal file not found: {args.normal_file}")
        if not args.reverse_file.exists():
            raise FileNotFoundError(f"Reverse file not found: {args.reverse_file}")

        output_file = args.output_file or (args.output_dir / args.normal_file.name)
        process_pair(args.normal_file, args.reverse_file, output_file)
        return

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    pairs = collect_pairs(args.input_dir)
    if not pairs:
        print("No normal/reverse file pairs found.")
        return

    total_kept = 0
    total_normal = 0
    total_position_dependent = 0

    for normal_file, reverse_file in pairs:
        output_file = args.output_dir / normal_file.name
        stats = process_pair(normal_file, reverse_file, output_file)
        total_kept += stats["kept"]
        total_normal += stats["normal_total"]
        total_position_dependent += stats["position_dependent"]

    print("\n=== Summary ===")
    print(f"pairs_processed={len(pairs)}")
    print(f"total_normal_records={total_normal}")
    print(f"total_kept_position_independent={total_kept}")
    print(f"total_position_dependent_removed={total_position_dependent}")


if __name__ == "__main__":
    main()
