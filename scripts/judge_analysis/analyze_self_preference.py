"""
Analyze self-preference metrics from position-independent judge outputs.

Metrics per file:
1) judge_accuracy:
    judge_output selects an answer whose output_label matches gold_label
2) self_selected_rate:
   judge_output == '1' (judge selects answer1/self) among all valid judge outputs
3) harmful_self_preference_rate:
    among samples where self model is wrong, proportion where judge still selects self

Default input directory:
  data/06_position_independent_response

Usage:
  python3 scripts/judge_analysis/analyze_self_preference.py
  python3 scripts/judge_analysis/analyze_self_preference.py --input-file data/06_position_independent_response/llama31_8.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "data" / "06_position_independent_response" / "raw"
)


def normalize_gold_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None

    if text in {"A", "B", "C", "D"}:
        return text

    if text in {"1", "2", "3", "4"}:
        return chr(ord("A") + int(text) - 1)

    return None


def normalize_answer_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    return None


def normalize_judge_output(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"1", "2"}:
        return text
    return None


def find_output_label_fields(record: dict[str, Any]) -> list[str]:
    return [k for k in record.keys() if k.endswith("_output_label")]


def analyze_file(path: Path) -> dict[str, Any]:
    total_records = 0

    judge_eval_total = 0
    judge_correct = 0

    judge_valid_total = 0
    self_selected = 0

    self_wrong_with_judge = 0
    self_wrong_and_self_selected = 0

    self_label_field = None
    other_label_field = None

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            total_records += 1

            if self_label_field is None:
                label_fields = find_output_label_fields(row)
                if len(label_fields) != 2:
                    raise ValueError(
                        f"{path.name}:{line_no} expected exactly 2 '*_output_label' fields, got {len(label_fields)}"
                    )
                self_label_field = label_fields[0]
                other_label_field = label_fields[1]

            gold = normalize_gold_label(row.get("gold_label"))
            self_label = normalize_answer_label(row.get(self_label_field))
            other_label = normalize_answer_label(row.get(other_label_field))

            is_self_correct = None
            if gold is not None and self_label is not None:
                is_self_correct = self_label == gold

            judge = normalize_judge_output(row.get("judge_output"))
            if judge is not None:
                judge_valid_total += 1

                selected_label = self_label if judge == "1" else other_label
                if gold is not None and selected_label is not None:
                    judge_eval_total += 1
                    if selected_label == gold:
                        judge_correct += 1

                if judge == "1":
                    self_selected += 1

                if is_self_correct is False:
                    self_wrong_with_judge += 1
                    if judge == "1":
                        self_wrong_and_self_selected += 1

    judge_accuracy = (judge_correct / judge_eval_total) if judge_eval_total else None
    self_selected_rate = (
        (self_selected / judge_valid_total) if judge_valid_total else None
    )
    harmful_self_pref_rate = (
        self_wrong_and_self_selected / self_wrong_with_judge
        if self_wrong_with_judge
        else None
    )

    return {
        "file": str(path),
        "file_name": path.name,
        "self_label_field": self_label_field,
        "other_label_field": other_label_field,
        "total_records": total_records,
        "judge_eval_total": judge_eval_total,
        "judge_correct": judge_correct,
        "judge_valid_total": judge_valid_total,
        "self_selected": self_selected,
        "self_wrong_with_judge": self_wrong_with_judge,
        "self_wrong_and_self_selected": self_wrong_and_self_selected,
        "judge_accuracy": judge_accuracy,
        "self_selected_rate": self_selected_rate,
        "harmful_self_preference_rate": harmful_self_pref_rate,
    }


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def collect_files(input_dir: Path, input_file: Path | None) -> list[Path]:
    if input_file is not None:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        return [input_file]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    return sorted([p for p in input_dir.glob("*.jsonl") if p.is_file()])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze judge accuracy and self-preference metrics for judge outputs."
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

    sum_judge_eval_total = sum(r["judge_eval_total"] for r in all_results)
    sum_judge_correct = sum(r["judge_correct"] for r in all_results)
    sum_judge_valid_total = sum(r["judge_valid_total"] for r in all_results)
    sum_self_selected = sum(r["self_selected"] for r in all_results)
    sum_self_wrong_with_judge = sum(r["self_wrong_with_judge"] for r in all_results)
    sum_self_wrong_and_self_selected = sum(
        r["self_wrong_and_self_selected"] for r in all_results
    )

    print("=== Per-file metrics ===")
    for r in all_results:
        print(f"\n[{r['file_name']}]")
        print(f"  self_label_field={r['self_label_field']}")
        print(f"  total_records={r['total_records']}")
        print(
            f"  judge_accuracy={fmt_pct(r['judge_accuracy'])} ({r['judge_correct']}/{r['judge_eval_total']})"
        )
        print(
            f"  self_selected_rate={fmt_pct(r['self_selected_rate'])} ({r['self_selected']}/{r['judge_valid_total']})"
        )
        print(
            "  "
            f"harmful_self_preference_rate={fmt_pct(r['harmful_self_preference_rate'])} "
            f"({r['self_wrong_and_self_selected']}/{r['self_wrong_with_judge']})"
        )

    overall_judge_accuracy = (
        (sum_judge_correct / sum_judge_eval_total) if sum_judge_eval_total else None
    )
    overall_self_selected_rate = (
        (sum_self_selected / sum_judge_valid_total) if sum_judge_valid_total else None
    )
    overall_harmful_rate = (
        sum_self_wrong_and_self_selected / sum_self_wrong_with_judge
        if sum_self_wrong_with_judge
        else None
    )

    print("\n=== Overall ===")
    print(f"files={len(all_results)}")
    print(
        f"judge_accuracy={fmt_pct(overall_judge_accuracy)} ({sum_judge_correct}/{sum_judge_eval_total})"
    )
    print(
        f"self_selected_rate={fmt_pct(overall_self_selected_rate)} ({sum_self_selected}/{sum_judge_valid_total})"
    )
    print(
        f"harmful_self_preference_rate={fmt_pct(overall_harmful_rate)} "
        f"({sum_self_wrong_and_self_selected}/{sum_self_wrong_with_judge})"
    )


if __name__ == "__main__":
    main()
