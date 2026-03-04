"""
Apply Gemini synonym replacements to the primary model's output_reason
in position-independent pairwise files, then strip judge fields.

Input:
  - target-dir: data/065_position_independent/before_restyle/
  - restyle-dir: data/085_restyle_batch_output/

Output:
  - output-dir: data/095_restyled_pairwise/

For each target file (e.g. llama33_70b__ds_v3__raw_judge.jsonl):
  1. Extract primary model name from filename (before first "__").
  2. Load synonym map from restyle-dir/{primary}.jsonl keyed by question_unique_id.
  3. For each record:
     - Apply word-level synonym replacement on {primary}_output_reason.
     - Remove judge_output, prob_1, prob_2 fields.
  4. Write to output-dir with same filename.

Usage:
  python scripts/answer_extraction/apply_restyle_to_pairwise.py
  python scripts/answer_extraction/apply_restyle_to_pairwise.py \
      --target-dir data/065_position_independent/before_restyle \
      --restyle-dir data/085_restyle_batch_output \
      --output-dir data/095_restyled_pairwise
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_TARGET_DIR = (
    PROJECT_ROOT / "data" / "065_position_independent" / "before_restyle"
)
DEFAULT_RESTYLE_DIR = PROJECT_ROOT / "data" / "085_restyle_batch_output"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "095_restyled_pairwise"

JUDGE_FIELDS_TO_REMOVE = {"judge_output", "prob_1", "prob_2"}


# ── Synonym parsing (reused from replace_answers_with_synonyms.py) ───


def parse_synonym_payload(payload_text: str) -> tuple[list[str], list[str]]:
    payload_text = payload_text.strip()
    if not payload_text:
        return [], []

    try:
        data = json.loads(payload_text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", payload_text)
        if not match:
            return [], []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], []

    if not isinstance(data, dict):
        return [], []

    selected = data.get("selected_words", [])
    replacements = data.get("replacements", [])

    if not isinstance(selected, list) or not isinstance(replacements, list):
        return [], []

    return [str(x) for x in selected], [str(x) for x in replacements]


def extract_synonyms_from_row(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    try:
        candidates = row["response"]["candidates"]
        if not candidates:
            return [], []
        parts = candidates[0]["content"]["parts"]
        if not parts:
            return [], []
        payload_text = parts[0].get("text", "")
        return parse_synonym_payload(str(payload_text))
    except Exception:
        return [], []


def apply_replacements(
    text: str, selected_words: list[str], replacement_words: list[str]
) -> str:
    result = text
    for old, new in zip(selected_words, replacement_words):
        old = old.strip()
        if not old:
            continue
        if re.fullmatch(r"[A-Za-z0-9_'-]+", old):
            pattern = re.compile(rf"\b{re.escape(old)}\b")
            result = pattern.sub(new, result)
        else:
            result = result.replace(old, new)
    return result


# ── Restyle map loading ──────────────────────────────────────────────


def load_restyle_map(
    restyle_file: Path,
) -> dict[str, tuple[list[str], list[str]]]:
    """Load question_unique_id → (selected_words, replacements) from 085 batch output."""
    mapping: dict[str, tuple[list[str], list[str]]] = {}

    with restyle_file.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error in {restyle_file} line {lineno}: {exc}"
                ) from exc

            qid = row.get("key")
            if not qid:
                continue

            selected, replacements = extract_synonyms_from_row(row)
            if selected and replacements:
                mapping[str(qid)] = (selected, replacements)

    return mapping


# ── File processing ──────────────────────────────────────────────────


def extract_primary_model(filename: str) -> str:
    """Extract primary model name from filename like llama33_70b__ds_v3__raw_judge.jsonl."""
    stem = filename.replace(".jsonl", "")
    return stem.split("__")[0]


def collect_target_files(target_dir: Path) -> list[Path]:
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    return sorted(p for p in target_dir.glob("*.jsonl") if p.is_file())


def find_restyle_file(primary: str, restyle_dir: Path) -> Path:
    path = restyle_dir / f"{primary}.jsonl"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Restyle file not found for model '{primary}': {path}"
    )


def process_file(
    target_file: Path,
    restyle_map: dict[str, tuple[list[str], list[str]]],
    output_file: Path,
    primary: str,
) -> dict[str, int]:
    reason_key = f"{primary}_output_reason"

    stats = {
        "total": 0,
        "replaced": 0,
        "no_restyle_match": 0,
        "missing_reason_key": 0,
        "empty_synonyms": 0,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with target_file.open("r", encoding="utf-8") as fin, output_file.open(
        "w", encoding="utf-8"
    ) as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error in {target_file} line {lineno}: {exc}"
                ) from exc

            qid = str(row.get("question_unique_id", ""))

            # Remove judge fields
            for field in JUDGE_FIELDS_TO_REMOVE:
                row.pop(field, None)

            if reason_key not in row:
                stats["missing_reason_key"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            synonyms = restyle_map.get(qid)
            if synonyms is None:
                stats["no_restyle_match"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            selected_words, replacement_words = synonyms
            if not selected_words:
                stats["empty_synonyms"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            original_reason = row[reason_key]
            row[reason_key] = apply_replacements(
                original_reason, selected_words, replacement_words
            )
            stats["replaced"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    return stats


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply synonym replacements to primary model's output_reason "
        "in pairwise files, strip judge fields, output to 095."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=f"Directory with position-independent pairwise files (default: {DEFAULT_TARGET_DIR})",
    )
    parser.add_argument(
        "--restyle-dir",
        type=Path,
        default=DEFAULT_RESTYLE_DIR,
        help=f"Directory with Gemini restyle batch outputs (default: {DEFAULT_RESTYLE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    target_files = collect_target_files(args.target_dir)
    if not target_files:
        print("No target files found.")
        return

    # Cache restyle maps per primary model (avoid reloading for same model)
    restyle_cache: dict[str, dict[str, tuple[list[str], list[str]]]] = {}

    total_rows = 0
    total_replaced = 0

    for target_file in target_files:
        primary = extract_primary_model(target_file.name)

        if primary not in restyle_cache:
            restyle_file = find_restyle_file(primary, args.restyle_dir)
            restyle_cache[primary] = load_restyle_map(restyle_file)
            print(
                f"[load] {restyle_file.name}: {len(restyle_cache[primary])} synonym entries"
            )

        restyle_map = restyle_cache[primary]
        output_file = args.output_dir / target_file.name

        stats = process_file(target_file, restyle_map, output_file, primary)
        total_rows += stats["total"]
        total_replaced += stats["replaced"]

        print(
            f"[done] {target_file.name}: "
            f"replaced={stats['replaced']}/{stats['total']} "
            f"no_restyle_match={stats['no_restyle_match']} "
            f"missing_reason_key={stats['missing_reason_key']} "
            f"empty_synonyms={stats['empty_synonyms']}"
        )

    print("\n=== Summary ===")
    print(f"files_processed={len(target_files)}")
    print(f"total_rows={total_rows}")
    print(f"total_replaced={total_replaced}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
