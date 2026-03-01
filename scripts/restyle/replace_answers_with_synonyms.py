"""
Apply Gemini synonym replacements to restyle outputs.

Input directory (default):
  data/09_gemini_restyle_output/with_ds

Output directory (default):
  data/10_replaced_answer/with_ds

For each input JSONL row, output only:
  - question_unique_id (from key)
  - Answer (answer text after synonym replacement)

Replacement behavior:
- Uses selected_words / replacements from Gemini response JSON.
- Applies full replacement (global) on the extracted answer text.
- Removes the Question section and keeps only content after "Answer:".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "09_gemini_restyle_output" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "10_replaced_answer" / "raw"


def collect_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(
        p
        for p in input_dir.glob("*.jsonl")
        if p.is_file() and not p.name.startswith("batch_")
    )
    return files


def extract_answer_block(user_text: str) -> str:
    # Keep everything after the first "Answer:" marker.
    match = re.search(r"Answer:\s*", user_text)
    if not match:
        return user_text.strip()
    return user_text[match.end() :].strip()


def parse_synonym_payload(payload_text: str) -> tuple[list[str], list[str]]:
    payload_text = payload_text.strip()
    if not payload_text:
        return [], []

    try:
        data = json.loads(payload_text)
    except json.JSONDecodeError:
        # Try extracting the first JSON object block.
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

    selected_words = [str(x) for x in selected]
    replacement_words = [str(x) for x in replacements]
    return selected_words, replacement_words


def apply_replacements(answer_text: str, selected_words: list[str], replacement_words: list[str]) -> str:
    result = answer_text

    for old, new in zip(selected_words, replacement_words):
        old = old.strip()
        if not old:
            continue

        # If old token is a single simple word, replace by word-boundary.
        if re.fullmatch(r"[A-Za-z0-9_'-]+", old):
            pattern = re.compile(rf"\b{re.escape(old)}\b")
            result = pattern.sub(new, result)
        else:
            # Phrase/punctuation case: plain global replace.
            result = result.replace(old, new)

    return result


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


def extract_user_text(row: dict[str, Any]) -> str:
    try:
        contents = row["request"]["contents"]
        if not contents:
            return ""
        parts = contents[0]["parts"]
        if not parts:
            return ""
        return str(parts[0].get("text", ""))
    except Exception:
        return ""


def process_file(input_file: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / input_file.name

    stats = {
        "total": 0,
        "written": 0,
        "missing_key": 0,
        "missing_answer": 0,
    }

    with input_file.open("r", encoding="utf-8") as fin, output_file.open(
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
                raise ValueError(f"JSON parse error in {input_file} line {lineno}: {exc}") from exc

            qid = row.get("key")
            if not qid:
                stats["missing_key"] += 1
                continue

            user_text = extract_user_text(row)
            answer_text = extract_answer_block(user_text)
            if not answer_text:
                stats["missing_answer"] += 1
                continue

            selected_words, replacement_words = extract_synonyms_from_row(row)
            replaced_answer = apply_replacements(answer_text, selected_words, replacement_words)

            out_row = {
                "question_unique_id": str(qid),
                "Answer": replaced_answer,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            stats["written"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply synonym replacements and output only question_unique_id + Answer."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    files = collect_input_files(args.input_dir)
    if not files:
        print("No input .jsonl files found.")
        return

    total_rows = 0
    total_written = 0

    for file_path in files:
        stats = process_file(file_path, args.output_dir)
        total_rows += stats["total"]
        total_written += stats["written"]
        print(
            f"[done] {file_path.name}: total={stats['total']} written={stats['written']} "
            f"missing_key={stats['missing_key']} missing_answer={stats['missing_answer']}"
        )

    print("\n=== Summary ===")
    print(f"files_processed={len(files)}")
    print(f"total_rows={total_rows}")
    print(f"total_written={total_written}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
