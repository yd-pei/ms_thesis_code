"""
Compute continuous self-preference scores from normal + swapped judge outputs.

Uses the paper's methodology:
  self_preference = (P_self_forward + P_self_backward) / 2
where:
  P_self_forward = prob_1 in normal file (self in position 1)
  P_self_backward = prob_2 in swapped file (self in position 2)

Reports per-pair and overall metrics split by beneficial/harmful quadrants.
No position-independence filtering is applied.

Default input directory:
  data/055_judge_output/

Usage:
  python scripts/judge_analysis/analyze_self_preference_continuous.py
  python scripts/judge_analysis/analyze_self_preference_continuous.py --input-dir data/055_judge_output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "055_judge_output"


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    if text in {"1", "2", "3", "4"}:
        return chr(ord("A") + int(text) - 1)
    return None


def find_label_fields(record: dict) -> list[str]:
    return [k for k in record.keys() if k.endswith("_output_label")]


def load_records(path: Path) -> dict[str, dict]:
    """Load JSONL, keyed by question_unique_id."""
    mapping: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_unique_id", ""))
            if qid:
                mapping[qid] = row
    return mapping


def discover_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Find (normal, swapped) file pairs in input_dir."""
    pairs = []
    for normal_file in sorted(input_dir.glob("*__raw_judge.jsonl")):
        if "_swapped" in normal_file.name:
            continue
        swapped_name = normal_file.name.replace("__raw_judge.jsonl", "__raw_judge_swapped.jsonl")
        swapped_file = input_dir / swapped_name
        if swapped_file.exists():
            pairs.append((normal_file, swapped_file))
        else:
            print(f"[warn] No swapped file for {normal_file.name}, skipping")
    return pairs


def extract_pair_name(normal_file: Path) -> tuple[str, str]:
    """Extract (primary, opponent) from filename like llama33_70b__ds_v3__raw_judge.jsonl."""
    stem = normal_file.stem  # llama33_70b__ds_v3__raw_judge
    parts = stem.split("__")
    return parts[0], parts[1]


def analyze_pair(
    normal_records: dict[str, dict],
    swapped_records: dict[str, dict],
    self_label_field: str,
    other_label_field: str,
) -> dict[str, Any]:
    common_qids = set(normal_records.keys()) & set(swapped_records.keys())

    stats = {
        "total_paired": len(common_qids),
        "ben_n": 0,
        "ben_self_pref_sum": 0.0,
        "harm_n": 0,
        "harm_self_pref_sum": 0.0,
        "fwd_correct": 0,
        "fwd_total": 0,
        "pos_independent": 0,
    }

    for qid in common_qids:
        nr = normal_records[qid]
        sr = swapped_records[qid]

        gold = normalize_label(nr.get("gold_label"))
        self_label = normalize_label(nr.get(self_label_field))
        other_label = normalize_label(nr.get(other_label_field))

        if gold is None or self_label is None or other_label is None:
            continue

        # Self-preference: prob_1 from normal (self in pos1), prob_2 from swapped (self in pos2)
        p_self_fwd = nr.get("prob_1")
        p_self_bwd = sr.get("prob_2")

        if p_self_fwd is None or p_self_bwd is None:
            continue

        self_pref = (p_self_fwd + p_self_bwd) / 2

        # Quadrant (XOR guaranteed by 045 filtering)
        self_correct = self_label == gold
        other_correct = other_label == gold

        if self_correct and not other_correct:
            stats["ben_n"] += 1
            stats["ben_self_pref_sum"] += self_pref
        elif not self_correct and other_correct:
            stats["harm_n"] += 1
            stats["harm_self_pref_sum"] += self_pref

        # Forward-only judge accuracy
        fwd_judge = str(nr.get("judge_output", "")).strip()
        if fwd_judge in {"1", "2"}:
            stats["fwd_total"] += 1
            chosen_label = self_label if fwd_judge == "1" else other_label
            if chosen_label == gold:
                stats["fwd_correct"] += 1

        # Position independence: forward and backward choose same MODEL
        bwd_judge = str(sr.get("judge_output", "")).strip()
        if fwd_judge in {"1", "2"} and bwd_judge in {"1", "2"}:
            # Forward "1" = chose self, backward "2" = chose self → consistent
            fwd_chose_self = fwd_judge == "1"
            bwd_chose_self = bwd_judge == "2"
            if fwd_chose_self == bwd_chose_self:
                stats["pos_independent"] += 1

    return stats


def fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "N/A"
    return f"{num / den * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute continuous self-preference scores (paper methodology)."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()

    pairs = discover_pairs(args.input_dir)
    if not pairs:
        print("No (normal, swapped) file pairs found.")
        return

    # Accumulate overall stats
    overall = {
        "ben_n": 0, "ben_self_pref_sum": 0.0,
        "harm_n": 0, "harm_self_pref_sum": 0.0,
        "fwd_correct": 0, "fwd_total": 0,
        "total_paired": 0, "pos_independent": 0,
    }

    print("=== Per-pair metrics ===")

    for normal_file, swapped_file in pairs:
        primary, opponent = extract_pair_name(normal_file)

        normal_records = load_records(normal_file)
        swapped_records = load_records(swapped_file)

        # Determine label fields from first record
        first_record = next(iter(normal_records.values()))
        label_fields = find_label_fields(first_record)
        if len(label_fields) != 2:
            print(f"[skip] {normal_file.name}: expected 2 label fields, got {len(label_fields)}")
            continue

        self_label_field = label_fields[0]  # primary model's label
        other_label_field = label_fields[1]

        stats = analyze_pair(normal_records, swapped_records, self_label_field, other_label_field)

        ben_avg = stats["ben_self_pref_sum"] / stats["ben_n"] if stats["ben_n"] else float("nan")
        harm_avg = stats["harm_self_pref_sum"] / stats["harm_n"] if stats["harm_n"] else float("nan")

        print(f"\n[{primary} vs {opponent}]  (paired: {stats['total_paired']})")
        print(f"  beneficial: n={stats['ben_n']:>4}  avg_self_pref={ben_avg:.4f}")
        print(f"  harmful:    n={stats['harm_n']:>4}  avg_self_pref={harm_avg:.4f}")
        print(f"  judge_accuracy (forward): {fmt_pct(stats['fwd_correct'], stats['fwd_total'])} ({stats['fwd_correct']}/{stats['fwd_total']})")
        print(f"  position_independent: {fmt_pct(stats['pos_independent'], stats['total_paired'])} ({stats['pos_independent']}/{stats['total_paired']})")

        for key in overall:
            overall[key] += stats[key]

    # Overall
    ben_avg_all = overall["ben_self_pref_sum"] / overall["ben_n"] if overall["ben_n"] else float("nan")
    harm_avg_all = overall["harm_self_pref_sum"] / overall["harm_n"] if overall["harm_n"] else float("nan")

    print("\n\n=== Overall ===")
    print(f"  total_paired: {overall['total_paired']}")
    print(f"  beneficial: n={overall['ben_n']:>4}  avg_self_pref={ben_avg_all:.4f}")
    print(f"  harmful:    n={overall['harm_n']:>4}  avg_self_pref={harm_avg_all:.4f}")
    print(f"  judge_accuracy (forward): {fmt_pct(overall['fwd_correct'], overall['fwd_total'])} ({overall['fwd_correct']}/{overall['fwd_total']})")
    print(f"  position_independent: {fmt_pct(overall['pos_independent'], overall['total_paired'])} ({overall['pos_independent']}/{overall['total_paired']})")


if __name__ == "__main__":
    main()
