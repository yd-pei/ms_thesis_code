"""
Compute accuracy statistics from standardized model output files.

Input JSONL schema (one record per line):
  {
    "question_unique_id": "...",
    "gold_label":   "A",   # ground truth letter
    "output_label": "B",   # model prediction; null = unparseable
    "reason":       "..."
  }

Usage:
    python scripts/eval_accuracy.py                        # all files in default dir
    python scripts/eval_accuracy.py data/02_quality_response/ds_v3.jsonl
    python scripts/eval_accuracy.py data/02_quality_response/*.jsonl --breakdown
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "02_quality_response"

LETTERS = ["A", "B", "C", "D"]


def evaluate(path: Path, breakdown: bool = False) -> dict:
    total = correct = skipped = 0
    per_gold: dict[str, list[bool]] = defaultdict(list)   # gold -> [correct?]
    per_pred: dict[str, int] = defaultdict(int)           # predicted letter -> count

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] {path.name}:{lineno}: {e}", file=sys.stderr)
                continue

            gold = record.get("gold_label")
            pred = record.get("output_label")

            if pred is None:
                skipped += 1
                continue

            total += 1
            is_correct = pred.upper() == gold.upper()
            if is_correct:
                correct += 1

            per_gold[gold.upper()].append(is_correct)
            per_pred[pred.upper()] += 1

    return {
        "total": total,
        "correct": correct,
        "skipped": skipped,
        "accuracy": correct / total if total else 0.0,
        "per_gold": dict(per_gold),
        "per_pred": dict(per_pred),
    }


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def print_report(path: Path, stats: dict, breakdown: bool) -> None:
    name = path.stem
    total, correct, skipped = stats["total"], stats["correct"], stats["skipped"]
    acc = stats["accuracy"]

    print(f"\n{'=' * 50}")
    print(f"  Model : {name}")
    print(f"{'=' * 50}")
    print(f"  Total evaluated : {total:>6}")
    print(f"  Correct         : {correct:>6}")
    print(f"  Skipped (null)  : {skipped:>6}")
    print(f"  Accuracy        : {fmt_pct(acc):>8}")

    if breakdown:
        print()
        print("  Per gold-label accuracy:")
        for letter in LETTERS:
            results = stats["per_gold"].get(letter, [])
            if not results:
                continue
            n = len(results)
            c = sum(results)
            print(f"    {letter}: {c:>4}/{n:<4}  {fmt_pct(c/n)}")

        print()
        print("  Prediction distribution:")
        total_pred = sum(stats["per_pred"].values())
        for letter in LETTERS:
            cnt = stats["per_pred"].get(letter, 0)
            bar = "#" * int(cnt / max(total_pred, 1) * 30)
            print(f"    {letter}: {cnt:>5}  {fmt_pct(cnt/total_pred if total_pred else 0)}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate accuracy of standardized model outputs.")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help=f"Input JSONL file(s). Defaults to all *.jsonl in {DEFAULT_DIR}",
    )
    parser.add_argument(
        "--breakdown", "-b",
        action="store_true",
        help="Show per-gold-label accuracy and prediction distribution.",
    )
    args = parser.parse_args()

    paths: list[Path] = args.files or sorted(DEFAULT_DIR.glob("*.jsonl"))
    if not paths:
        print(f"[error] No JSONL files found in {DEFAULT_DIR}", file=sys.stderr)
        sys.exit(1)

    all_stats: list[tuple[str, dict]] = []
    for path in paths:
        path = path.expanduser().resolve()
        if not path.exists():
            print(f"[error] File not found: {path}", file=sys.stderr)
            continue
        stats = evaluate(path, breakdown=args.breakdown)
        print_report(path, stats, args.breakdown)
        all_stats.append((path.stem, stats))

    # Summary table when multiple files
    if len(all_stats) > 1:
        print(f"\n{'=' * 50}")
        print("  Summary")
        print(f"{'=' * 50}")
        col = max(len(n) for n, _ in all_stats)
        header = f"  {'Model':<{col}}  {'Total':>6}  {'Correct':>7}  {'Skipped':>7}  {'Accuracy':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, s in sorted(all_stats, key=lambda x: -x[1]["accuracy"]):
            print(
                f"  {name:<{col}}  {s['total']:>6}  {s['correct']:>7}  "
                f"{s['skipped']:>7}  {fmt_pct(s['accuracy']):>9}"
            )
        print()


if __name__ == "__main__":
    main()
