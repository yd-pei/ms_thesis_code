"""
Replace model output reasons in position-independent files using restyled answers.

Default input target directory:
  data/06_position_independent_response/greedy

Default replacement source directory:
  data/10_replaced_answer/with_ds

Behavior:
- Match rows one-to-one by question_unique_id.
- Only replace <model>_output_reason field with source "Answer".
- Keep all other fields unchanged (gold_label, output_label, ds_v3 fields, etc.).
- By default writes back in place to the target directory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_TARGET_DIR = PROJECT_ROOT / "data" / "06_position_independent_response" / "greedy"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "10_replaced_answer" / "with_ds"


def collect_target_files(target_dir: Path) -> list[Path]:
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    return sorted([p for p in target_dir.glob("*_judge.jsonl") if p.is_file()])


def infer_model_prefix(target_file: Path) -> str:
    stem = target_file.stem
    if not stem.endswith("_judge"):
        raise ValueError(f"Unexpected target filename: {target_file.name}")
    return stem[: -len("_judge")]


def model_prefix_to_source_candidates(model_prefix: str) -> list[str]:
    candidates = [f"{model_prefix}.jsonl"]

    # e.g., llama31_8b -> llama_31_8b
    m = re.fullmatch(r"llama(\d+)_([0-9]+b)", model_prefix)
    if m:
        family, size = m.groups()
        candidates.insert(0, f"llama_{family}_{size}.jsonl")

    # e.g., llama31_70b -> llama_31_70b
    m2 = re.fullmatch(r"llama(\d+)_([0-9]+b)", model_prefix)
    if m2:
        family, size = m2.groups()
        alt = f"llama_{family}_{size}.jsonl"
        if alt not in candidates:
            candidates.insert(0, alt)

    return candidates


def find_source_file(model_prefix: str, source_dir: Path) -> Path:
    for name in model_prefix_to_source_candidates(model_prefix):
        path = source_dir / name
        if path.exists():
            return path
    tried = ", ".join(model_prefix_to_source_candidates(model_prefix))
    raise FileNotFoundError(
        f"No replacement source file for model '{model_prefix}' in {source_dir}. Tried: {tried}"
    )


def load_answer_map(source_file: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with source_file.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error in {source_file} line {lineno}: {exc}") from exc

            qid = row.get("question_unique_id")
            answer = row.get("Answer")
            if not qid or answer is None:
                continue
            mapping[str(qid)] = str(answer)
    return mapping


def process_file(target_file: Path, source_file: Path, output_file: Path, model_prefix: str) -> dict[str, int]:
    answer_map = load_answer_map(source_file)
    reason_key = f"{model_prefix}_output_reason"

    stats = {
        "total": 0,
        "replaced": 0,
        "missing_qid": 0,
        "missing_answer": 0,
        "missing_reason_key": 0,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    in_place = target_file.resolve() == output_file.resolve()
    write_path = output_file.with_suffix(output_file.suffix + ".tmp") if in_place else output_file

    with target_file.open("r", encoding="utf-8") as fin, write_path.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error in {target_file} line {lineno}: {exc}") from exc

            qid = row.get("question_unique_id")
            if not qid:
                stats["missing_qid"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            if reason_key not in row:
                stats["missing_reason_key"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            new_answer = answer_map.get(str(qid))
            if new_answer is None:
                stats["missing_answer"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            row[reason_key] = new_answer
            stats["replaced"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    if in_place:
        write_path.replace(output_file)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace <model>_output_reason in 06_position_independent_response by question_unique_id."
    )
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for updated files. Default: overwrite target-dir in place.",
    )
    args = parser.parse_args()

    target_files = collect_target_files(args.target_dir)
    if not target_files:
        print("No target files found.")
        return

    out_dir = args.output_dir or args.target_dir

    total_rows = 0
    total_replaced = 0

    for target_file in target_files:
        model_prefix = infer_model_prefix(target_file)
        source_file = find_source_file(model_prefix, args.source_dir)
        output_file = out_dir / target_file.name

        stats = process_file(
            target_file=target_file,
            source_file=source_file,
            output_file=output_file,
            model_prefix=model_prefix,
        )

        total_rows += stats["total"]
        total_replaced += stats["replaced"]

        print(
            f"[done] {target_file.name}: replaced={stats['replaced']}/{stats['total']} "
            f"missing_answer={stats['missing_answer']} missing_qid={stats['missing_qid']} "
            f"missing_reason_key={stats['missing_reason_key']} src={source_file.name}"
        )

    print("\n=== Summary ===")
    print(f"files_processed={len(target_files)}")
    print(f"total_rows={total_rows}")
    print(f"total_replaced={total_replaced}")
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
