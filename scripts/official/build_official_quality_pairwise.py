"""
Build official QuALITY pairwise JSONL inputs for current judge/raw_judge pipeline.

Outputs:
  - data/official/quality_official_context.jsonl
  - data/official/before_restyle/*.jsonl
  - data/official/after_restyle/*.jsonl
  - data/official/manifest.json

Rules:
  - Keep only rows where exactly one side is correct (XOR).
  - Keep only two judge aliases by default: llama31_8b,qwen25_7b.
  - Append label after reason text: "\\n\\nFinal Answer: X".
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_SOURCE_CSV = PROJECT_ROOT / "mitigatingselfpreference" / "ds" / "quality_responses.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "official"
DEFAULT_JUDGES = "llama31_8b,qwen25_7b"
DEFAULT_FILTER = "xor"
DEFAULT_LABEL_FORMAT = "\n\nFinal Answer: {label}"

MODEL_MAP: dict[str, str] = {
    "Meta-Llama-3.1-8B-Instruct-Turbo": "llama31_8b",
    "Qwen2.5-7B-Instruct-Turbo": "qwen25_7b",
    "DeepSeek-V3": "ds_v3",
    "Llama-4-Scout-17B-16E-Instruct": "llama4_scout_17b",
    "Llama-4-Maverick-17B-128E-Instruct-FP8": "llama4_maverick_17b",
    "Qwen2.5-72B-Instruct-Turbo": "qwen25_72b",
}

PID_PATTERN = re.compile(r"\d+_[A-Za-z0-9]+_\d+_\d+")


@dataclass
class PairBuildResult:
    before_row: dict[str, Any] | None
    after_row: dict[str, Any] | None
    drop_reason: str | None
    judge_reason_changed: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build official QuALITY pairwise JSONL files for judge/raw_judge "
            "from mitigatingselfpreference/ds/quality_responses.csv."
        )
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=DEFAULT_SOURCE_CSV,
        help=f"Official source CSV path (default: {DEFAULT_SOURCE_CSV})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--judges",
        type=str,
        default=DEFAULT_JUDGES,
        help=f"Comma-separated judge aliases (default: {DEFAULT_JUDGES})",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=DEFAULT_FILTER,
        help="Filtering mode, fixed to xor",
    )
    parser.add_argument(
        "--label-after-reason-format",
        type=str,
        default=DEFAULT_LABEL_FORMAT,
        help=r'Label suffix format (default: "\n\nFinal Answer: {label}")',
    )
    return parser.parse_args()


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in {"NULL", "NONE", "NAN", "N/A"}:
        return None
    if text in {"A", "B", "C", "D"}:
        return text

    patterns = [
        r"FINAL\s*ANSWER\s*[:\-]?\s*([ABCD])",
        r"OPTION\s*([ABCD])",
        r"\(([ABCD])\)",
        r"\b([ABCD])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def append_label_to_reason(reason_text: str, label: str, suffix_format: str) -> str:
    return reason_text.rstrip() + suffix_format.format(label=label)


def read_rows_with_valid_pid(
    source_csv: Path,
) -> tuple[dict[str, dict[str, Any]], Counter]:
    counters: Counter = Counter()
    rows_by_pid: dict[str, dict[str, Any]] = {}

    with source_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counters["rows_read"] += 1
            pid_raw = normalize_text(row.get("pid"))
            if not pid_raw:
                counters["dropped_missing_pid"] += 1
                continue
            if not PID_PATTERN.fullmatch(pid_raw):
                counters["dropped_invalid_pid_format"] += 1
                continue
            if pid_raw in rows_by_pid:
                counters["dropped_duplicate_pid"] += 1
                continue
            rows_by_pid[pid_raw] = row
            counters["rows_with_valid_pid"] += 1

    counters["unique_pid_count"] = len(rows_by_pid)
    return rows_by_pid, counters


def build_pair_rows(
    row: dict[str, Any],
    pid: str,
    judge_alias: str,
    opponent_alias: str,
    alias_to_model: dict[str, str],
    label_suffix_format: str,
) -> PairBuildResult:
    judge_model = alias_to_model[judge_alias]
    opponent_model = alias_to_model[opponent_alias]

    gold_label = normalize_label(row.get("output_label"))
    if gold_label is None:
        return PairBuildResult(None, None, "missing_or_invalid_gold_label")

    article = normalize_text(row.get("text"))
    if not article:
        return PairBuildResult(None, None, "missing_article")

    question = normalize_text(row.get("questions"))
    if not question:
        return PairBuildResult(None, None, "missing_question")

    judge_label = normalize_label(row.get(f"{judge_model}_output_label"))
    if judge_label is None:
        return PairBuildResult(None, None, "missing_or_invalid_judge_label")

    opponent_label = normalize_label(row.get(f"{opponent_model}_output_label"))
    if opponent_label is None:
        return PairBuildResult(None, None, "missing_or_invalid_opponent_label")

    judge_reason_before = normalize_text(row.get(f"{judge_model}_reason"))
    if not judge_reason_before:
        return PairBuildResult(None, None, "missing_judge_reason_before")

    judge_reason_after = normalize_text(row.get(f"{judge_model}_reason_perturb_llm_auto"))
    if not judge_reason_after:
        return PairBuildResult(None, None, "missing_judge_reason_after")

    opponent_reason = normalize_text(row.get(f"{opponent_model}_reason"))
    if not opponent_reason:
        return PairBuildResult(None, None, "missing_opponent_reason")

    judge_is_correct = judge_label == gold_label
    opponent_is_correct = opponent_label == gold_label
    if not (judge_is_correct ^ opponent_is_correct):
        return PairBuildResult(None, None, "not_xor")

    before_row = {
        "question_unique_id": pid,
        "gold_label": gold_label,
        "judge_model": judge_alias,
        "opponent_model": opponent_alias,
        f"{judge_alias}_output_label": judge_label,
        f"{judge_alias}_output_reason": append_label_to_reason(
            judge_reason_before, judge_label, label_suffix_format
        ),
        "opponent_output_label": opponent_label,
        "opponent_output_reason": append_label_to_reason(
            opponent_reason, opponent_label, label_suffix_format
        ),
    }

    after_row = {
        "question_unique_id": pid,
        "gold_label": gold_label,
        "judge_model": judge_alias,
        "opponent_model": opponent_alias,
        f"{judge_alias}_output_label": judge_label,
        f"{judge_alias}_output_reason": append_label_to_reason(
            judge_reason_after, judge_label, label_suffix_format
        ),
        "opponent_output_label": opponent_label,
        "opponent_output_reason": append_label_to_reason(
            opponent_reason, opponent_label, label_suffix_format
        ),
    }

    return PairBuildResult(
        before_row=before_row,
        after_row=after_row,
        drop_reason=None,
        judge_reason_changed=before_row[f"{judge_alias}_output_reason"]
        != after_row[f"{judge_alias}_output_reason"],
    )


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_qids(path: Path) -> list[str]:
    qids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_unique_id")
            if qid:
                qids.append(str(qid))
    return qids


def validate_pairwise_file(path: Path) -> dict[str, Any]:
    reason_fields_first: list[str] | None = None
    row_count = 0
    xor_ok_count = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row_count += 1
            row = json.loads(line)
            reason_fields = [k for k in row.keys() if k.endswith("_output_reason")]
            if len(reason_fields) != 2:
                raise ValueError(
                    f"{path} line {lineno}: expected exactly 2 '*_output_reason' fields, got {len(reason_fields)}"
                )
            if reason_fields_first is None:
                reason_fields_first = reason_fields
            elif reason_fields != reason_fields_first:
                raise ValueError(
                    f"{path} line {lineno}: inconsistent reason fields order, "
                    f"expected {reason_fields_first}, got {reason_fields}"
                )

            gold = normalize_label(row.get("gold_label"))
            judge_label_field = [k for k in row if k.endswith("_output_label") and k != "opponent_output_label"]
            if len(judge_label_field) != 1:
                raise ValueError(
                    f"{path} line {lineno}: expected exactly 1 judge label field, got {judge_label_field}"
                )
            judge_label = normalize_label(row.get(judge_label_field[0]))
            opponent_label = normalize_label(row.get("opponent_output_label"))
            if gold is None or judge_label is None or opponent_label is None:
                raise ValueError(f"{path} line {lineno}: invalid/missing label in validation")
            if (judge_label == gold) ^ (opponent_label == gold):
                xor_ok_count += 1

    return {
        "rows": row_count,
        "reason_fields": reason_fields_first or [],
        "xor_ok_rows": xor_ok_count,
        "xor_all_ok": xor_ok_count == row_count,
    }


def main() -> None:
    args = parse_args()

    if args.filter.lower() != "xor":
        raise ValueError(f"Only --filter xor is supported. Got: {args.filter}")
    if "{label}" not in args.label_after_reason_format:
        raise ValueError("--label-after-reason-format must contain '{label}' placeholder")
    if not args.source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {args.source_csv}")

    judges = [item.strip() for item in args.judges.split(",") if item.strip()]
    if not judges:
        raise ValueError("No judge aliases provided")

    alias_to_model = {alias: model for model, alias in MODEL_MAP.items()}
    unknown = sorted(set(judges) - set(alias_to_model.keys()))
    if unknown:
        raise ValueError(f"Unknown judge alias(es): {unknown}. Supported: {sorted(alias_to_model.keys())}")

    output_root: Path = args.output_root
    before_dir = output_root / "before_restyle"
    after_dir = output_root / "after_restyle"
    output_root.mkdir(parents=True, exist_ok=True)
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    rows_by_pid, row_counters = read_rows_with_valid_pid(args.source_csv)
    sorted_pids = sorted(rows_by_pid.keys())

    used_qids: set[str] = set()
    context_candidate_qids: set[str] = set()

    per_pair_stats: dict[str, dict[str, Any]] = {}
    per_judge_mix_before: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_judge_mix_after: dict[str, list[dict[str, Any]]] = defaultdict(list)
    judge_drop_counters: dict[str, Counter] = defaultdict(Counter)

    all_aliases = list(alias_to_model.keys())

    for judge in judges:
        opponents = [alias for alias in all_aliases if alias != judge]
        for opponent in opponents:
            pair_key = f"{judge}__{opponent}"
            pair_before_rows: list[dict[str, Any]] = []
            pair_after_rows: list[dict[str, Any]] = []
            drop_counter: Counter = Counter()
            changed_count = 0
            unchanged_count = 0

            for pid in sorted_pids:
                row = rows_by_pid[pid]
                result = build_pair_rows(
                    row=row,
                    pid=pid,
                    judge_alias=judge,
                    opponent_alias=opponent,
                    alias_to_model=alias_to_model,
                    label_suffix_format=args.label_after_reason_format,
                )

                if result.drop_reason:
                    drop_counter[result.drop_reason] += 1
                    judge_drop_counters[judge][result.drop_reason] += 1
                    continue

                assert result.before_row is not None
                assert result.after_row is not None

                pair_before_rows.append(result.before_row)
                pair_after_rows.append(result.after_row)
                per_judge_mix_before[judge].append(result.before_row)
                per_judge_mix_after[judge].append(result.after_row)
                used_qids.add(pid)

                context_candidate_qids.add(pid)

                if result.judge_reason_changed:
                    changed_count += 1
                else:
                    unchanged_count += 1

            before_path = before_dir / f"{pair_key}.jsonl"
            after_path = after_dir / f"{pair_key}.jsonl"
            write_jsonl(pair_before_rows, before_path)
            write_jsonl(pair_after_rows, after_path)

            before_qids = [row["question_unique_id"] for row in pair_before_rows]
            after_qids = [row["question_unique_id"] for row in pair_after_rows]

            per_pair_stats[pair_key] = {
                "judge": judge,
                "opponent": opponent,
                "before_file": str(before_path),
                "after_file": str(after_path),
                "before_rows": len(pair_before_rows),
                "after_rows": len(pair_after_rows),
                "before_after_qids_match": sorted(before_qids) == sorted(after_qids),
                "judge_reason_changed": changed_count,
                "judge_reason_unchanged": unchanged_count,
                "drop_reasons": dict(drop_counter),
            }

    mix_stats: dict[str, dict[str, Any]] = {}
    for judge in judges:
        mix_before_path = before_dir / f"{judge}__official_mix.jsonl"
        mix_after_path = after_dir / f"{judge}__official_mix.jsonl"
        mix_before_rows = per_judge_mix_before[judge]
        mix_after_rows = per_judge_mix_after[judge]
        write_jsonl(mix_before_rows, mix_before_path)
        write_jsonl(mix_after_rows, mix_after_path)

        mix_before_qids = [row["question_unique_id"] for row in mix_before_rows]
        mix_after_qids = [row["question_unique_id"] for row in mix_after_rows]
        mix_stats[judge] = {
            "before_file": str(mix_before_path),
            "after_file": str(mix_after_path),
            "before_rows": len(mix_before_rows),
            "after_rows": len(mix_after_rows),
            "before_after_qids_match": sorted(mix_before_qids) == sorted(mix_after_qids),
            "drop_reasons_aggregate": dict(judge_drop_counters[judge]),
        }

    context_rows: list[dict[str, Any]] = []
    for pid in sorted(context_candidate_qids):
        row = rows_by_pid.get(pid)
        if row is None:
            continue
        article = normalize_text(row.get("text"))
        question = normalize_text(row.get("questions"))
        if not article or not question:
            continue
        context_rows.append(
            {
                "question_unique_id": pid,
                "article": article,
                "question": question,
            }
        )

    context_path = output_root / "quality_official_context.jsonl"
    write_jsonl(context_rows, context_path)
    context_qids = {row["question_unique_id"] for row in context_rows}

    validation_summary: dict[str, dict[str, Any]] = {}
    files_to_validate: list[Path] = []
    for judge in judges:
        files_to_validate.append(before_dir / f"{judge}__official_mix.jsonl")
        files_to_validate.append(after_dir / f"{judge}__official_mix.jsonl")
        for opponent in all_aliases:
            if opponent == judge:
                continue
            files_to_validate.append(before_dir / f"{judge}__{opponent}.jsonl")
            files_to_validate.append(after_dir / f"{judge}__{opponent}.jsonl")

    for path in files_to_validate:
        result = validate_pairwise_file(path)
        file_qids = load_qids(path)
        missing_qids = sorted(set(file_qids) - context_qids)
        result["context_hit"] = len(missing_qids) == 0
        result["missing_context_qids"] = missing_qids[:20]
        validation_summary[str(path)] = result

    before_after_consistency: dict[str, Any] = {}
    for judge in judges:
        pair_entries: dict[str, Any] = {}
        before_mix_qids = set(load_qids(before_dir / f"{judge}__official_mix.jsonl"))
        after_mix_qids = set(load_qids(after_dir / f"{judge}__official_mix.jsonl"))
        pair_entries["official_mix"] = {
            "before_qids": len(before_mix_qids),
            "after_qids": len(after_mix_qids),
            "qid_sets_equal": before_mix_qids == after_mix_qids,
        }
        for opponent in all_aliases:
            if opponent == judge:
                continue
            before_file = before_dir / f"{judge}__{opponent}.jsonl"
            after_file = after_dir / f"{judge}__{opponent}.jsonl"
            before_qids = set(load_qids(before_file))
            after_qids = set(load_qids(after_file))
            pair_entries[f"{judge}__{opponent}"] = {
                "before_qids": len(before_qids),
                "after_qids": len(after_qids),
                "qid_sets_equal": before_qids == after_qids,
            }
        before_after_consistency[judge] = pair_entries

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(args.source_csv),
        "output_root": str(output_root),
        "filter": "xor",
        "label_after_reason_format": args.label_after_reason_format,
        "judges": judges,
        "model_mapping": MODEL_MAP,
        "source_row_stats": dict(row_counters),
        "context": {
            "file": str(context_path),
            "rows": len(context_rows),
            "unique_qids": len(context_qids),
            "used_qids": len(used_qids),
            "used_qids_all_in_context": len(used_qids - context_qids) == 0,
            "missing_used_qids": sorted(used_qids - context_qids)[:20],
        },
        "mix_stats": mix_stats,
        "pair_stats": per_pair_stats,
        "before_after_consistency": before_after_consistency,
        "validation": validation_summary,
    }

    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote context: {context_path} ({len(context_rows)} rows)")
    print(f"Wrote before files: {before_dir}")
    print(f"Wrote after files:  {after_dir}")
    print(f"Wrote manifest:    {manifest_path}")


if __name__ == "__main__":
    main()
