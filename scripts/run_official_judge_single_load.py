#!/usr/bin/env python3
"""
Run official judge/raw_judge with a single model load per judge alias.

This script does NOT modify input data files. It builds temporary merged inputs
with metadata, runs inference once per judge/model, then splits outputs back to
the same per-file naming scheme used by scripts/run_official_judge.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from align_lab.inference.judge import _find_reason_fields, run_vllm_judge_inference
from align_lab.inference.raw_judge import run_raw_judge_inference


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_MAP = {
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
}

PHASE_TO_INPUT_DIR = {
    "before": "before_restyle",
    "after": "after_restyle",
}

PHASE_TO_OUTPUT_DIR = {
    "before": "judged_before",
    "after": "judged_after",
}

META_PHASE = "__meta_phase"
META_INPUT_FILE = "__meta_input_file"
META_PASS = "__meta_pass"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official judge/raw_judge with one model load per judge alias."
    )
    parser.add_argument(
        "--mode",
        choices=["raw_judge", "judge", "both"],
        default="raw_judge",
        help="Inference mode (default: raw_judge).",
    )
    parser.add_argument(
        "--phase",
        choices=["before", "after", "both"],
        default="both",
        help="Data phase to run (default: both).",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "official",
        help="Root directory containing before_restyle/after_restyle.",
    )
    parser.add_argument(
        "--quality-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "official" / "quality_official_context.jsonl",
        help="Context JSONL used by judge/raw_judge.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "official",
        help="Root directory for judged_before/judged_after.",
    )
    parser.add_argument(
        "--judges",
        type=str,
        default="llama31_8b,qwen25_7b",
        help="Comma-separated judge aliases (default: llama31_8b,qwen25_7b).",
    )
    parser.add_argument(
        "--with-swap",
        action="store_true",
        help="Also run swapped ordering, without additional model loads.",
    )
    parser.add_argument(
        "--raw-batch-size",
        type=int,
        default=16,
        help="Batch size for raw_judge (default: 16).",
    )
    parser.add_argument(
        "--judge-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "judge.yaml",
        help="Sampling config for judge mode.",
    )
    parser.add_argument(
        "--judge-quantization",
        choices=["bitsandbytes", "gptq", "awq", "fp8", "none"],
        default="none",
        help="Quantization mode for judge (default: none).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HF token. Defaults to HF_TOKEN from environment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions only.",
    )
    return parser.parse_args()


def resolve_phases(phase: str) -> list[str]:
    if phase == "both":
        return ["before", "after"]
    return [phase]


def resolve_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["raw_judge", "judge"]
    return [mode]


def load_sampling_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Judge config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Judge config must be a dict: {path}")
    return data


def iter_input_files(input_root: Path, phases: list[str], judge_alias: str) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for phase in phases:
        phase_dir = input_root / PHASE_TO_INPUT_DIR[phase]
        if not phase_dir.exists():
            continue
        for path in sorted(phase_dir.glob(f"{judge_alias}__*.jsonl")):
            if path.is_file():
                pairs.append((phase, path))
    return pairs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error in {path} line {lineno}: {exc}") from exc
    return rows


def write_merged_input(
    sources: list[tuple[str, Path]],
    merged_path: Path,
    with_swap: bool,
) -> dict[str, int]:
    stats = {
        "source_files": len(sources),
        "source_rows": 0,
        "merged_rows": 0,
    }
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    with merged_path.open("w", encoding="utf-8") as out_f:
        for phase, src_path in sources:
            rows = read_jsonl(src_path)
            for row in rows:
                reason_fields = _find_reason_fields(row)
                if len(reason_fields) != 2:
                    raise ValueError(
                        f"{src_path}: expected exactly 2 '*_output_reason' fields, got {len(reason_fields)}"
                    )
                reason_key_1, reason_key_2 = reason_fields
                stats["source_rows"] += 1

                normal_row = dict(row)
                normal_row[META_PHASE] = phase
                normal_row[META_INPUT_FILE] = src_path.name
                normal_row[META_PASS] = "normal"
                out_f.write(json.dumps(normal_row, ensure_ascii=False) + "\n")
                stats["merged_rows"] += 1

                if with_swap:
                    swapped_row = dict(row)
                    swapped_row[reason_key_1], swapped_row[reason_key_2] = (
                        swapped_row.get(reason_key_2),
                        swapped_row.get(reason_key_1),
                    )
                    swapped_row[META_PHASE] = phase
                    swapped_row[META_INPUT_FILE] = src_path.name
                    swapped_row[META_PASS] = "swapped"
                    out_f.write(json.dumps(swapped_row, ensure_ascii=False) + "\n")
                    stats["merged_rows"] += 1

    return stats


def split_merged_output(
    merged_output_path: Path,
    output_root: Path,
    mode: str,
) -> dict[str, int]:
    if mode == "raw_judge":
        suffix_base = "__raw_judge"
    elif mode == "judge":
        suffix_base = "__judge"
    else:
        raise ValueError(f"Unsupported mode for split: {mode}")

    file_handles: dict[Path, Any] = {}
    rows_written = 0
    files_written: set[Path] = set()

    try:
        with merged_output_path.open("r", encoding="utf-8") as in_f:
            for lineno, line in enumerate(in_f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON parse error in merged output {merged_output_path} line {lineno}: {exc}"
                    ) from exc

                phase = row.pop(META_PHASE, None)
                input_file = row.pop(META_INPUT_FILE, None)
                run_pass = row.pop(META_PASS, None)

                if phase not in PHASE_TO_OUTPUT_DIR:
                    raise ValueError(f"Missing/invalid phase metadata in merged output line {lineno}: {phase}")
                if not input_file or not input_file.endswith(".jsonl"):
                    raise ValueError(
                        f"Missing/invalid input file metadata in merged output line {lineno}: {input_file}"
                    )
                if run_pass not in {"normal", "swapped"}:
                    raise ValueError(
                        f"Missing/invalid pass metadata in merged output line {lineno}: {run_pass}"
                    )

                base_name = input_file[:-6]
                suffix = suffix_base if run_pass == "normal" else f"{suffix_base}_swapped"
                out_name = f"{base_name}{suffix}.jsonl"
                out_dir = output_root / PHASE_TO_OUTPUT_DIR[phase]
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / out_name

                if out_path not in file_handles:
                    file_handles[out_path] = out_path.open("w", encoding="utf-8")
                    files_written.add(out_path)
                file_handles[out_path].write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1
    finally:
        for handle in file_handles.values():
            handle.close()

    return {
        "rows_written": rows_written,
        "files_written": len(files_written),
    }


def main() -> None:
    load_dotenv()
    args = parse_args()

    input_root = args.input_root.resolve()
    quality_path = args.quality_path.resolve()
    output_root = args.output_root.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not quality_path.exists():
        raise FileNotFoundError(f"Quality path not found: {quality_path}")

    judges = [x.strip() for x in args.judges.split(",") if x.strip()]
    if not judges:
        raise ValueError("No judge aliases provided.")
    unknown = sorted(set(judges) - set(MODEL_MAP.keys()))
    if unknown:
        raise ValueError(f"Unknown judge alias(es): {unknown}. Supported: {sorted(MODEL_MAP.keys())}")

    phases = resolve_phases(args.phase)
    modes = resolve_modes(args.mode)
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    judge_sampling = load_sampling_config(args.judge_config) if "judge" in modes else {}
    quantization = None if args.judge_quantization == "none" else args.judge_quantization

    print("=== Official Single-Load Runner ===")
    print(f"input_root:   {input_root}")
    print(f"quality_path: {quality_path}")
    print(f"output_root:  {output_root}")
    print(f"judges:       {judges}")
    print(f"modes:        {modes}")
    print(f"phases:       {phases}")
    print(f"with_swap:    {args.with_swap}")
    print(f"raw_batch:    {args.raw_batch_size}")
    print(f"judge_quant:  {args.judge_quantization}")
    print(f"dry_run:      {args.dry_run}")

    for judge_alias in judges:
        model_path = MODEL_MAP[judge_alias]
        sources = iter_input_files(input_root=input_root, phases=phases, judge_alias=judge_alias)
        if not sources:
            print(f"[skip] {judge_alias}: no input files found.")
            continue

        print(f"\n[{judge_alias}] files={len(sources)} model={model_path}")

        with tempfile.TemporaryDirectory(prefix=f"official_single_load_{judge_alias}_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            merged_input_path = tmp_dir_path / f"{judge_alias}__merged_input.jsonl"
            merge_stats = write_merged_input(
                sources=sources,
                merged_path=merged_input_path,
                with_swap=args.with_swap,
            )
            print(
                f"[{judge_alias}] merged source_rows={merge_stats['source_rows']} "
                f"-> merged_rows={merge_stats['merged_rows']} at {merged_input_path}"
            )

            if args.dry_run:
                continue

            for mode in modes:
                merged_output_path = tmp_dir_path / f"{judge_alias}__{mode}_merged_output.jsonl"

                if mode == "raw_judge":
                    print(f"[{judge_alias}] running raw_judge once...")
                    run_raw_judge_inference(
                        model_path=model_path,
                        data_path=str(merged_input_path),
                        output_path=str(merged_output_path),
                        quality_path=str(quality_path),
                        swap_answers=False,
                        hf_token=hf_token,
                        batch_size=args.raw_batch_size,
                    )
                elif mode == "judge":
                    print(f"[{judge_alias}] running judge once...")
                    run_vllm_judge_inference(
                        model_path=model_path,
                        data_path=str(merged_input_path),
                        output_path=str(merged_output_path),
                        quality_path=str(quality_path),
                        swap_answers=False,
                        hf_token=hf_token,
                        quantization=quantization,
                        sampling_config=judge_sampling,
                    )
                else:
                    raise ValueError(f"Unsupported mode: {mode}")

                split_stats = split_merged_output(
                    merged_output_path=merged_output_path,
                    output_root=output_root,
                    mode=mode,
                )
                print(
                    f"[{judge_alias}] {mode} split rows={split_stats['rows_written']} "
                    f"files={split_stats['files_written']}"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()

