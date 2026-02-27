"""
Upload Gemini Batch JSONL files to Gemini Files API and record uploaded file metadata.

Default input directory:
  data/08_gemini_api/with_ds

Default markdown output:
  doc/restyle/gemini_file_api.md

Usage:
  uv run python scripts/restyle/upload2fileAPI.py
  uv run python scripts/restyle/upload2fileAPI.py --dry-run
  uv run python scripts/restyle/upload2fileAPI.py --append-md
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "08_gemini_api" / "with_ds"
DEFAULT_MD_PATH = PROJECT_ROOT / "doc" / "restyle" / "gemini_file_api.md"


load_dotenv(PROJECT_ROOT / ".env")


def get_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing API key. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in environment or .env."
        )
    return api_key


def get_attr(obj: Any, attr: str, default: str = "") -> str:
    value = getattr(obj, attr, None)
    if value is None:
        return default
    return str(value)


def collect_jsonl_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return sorted([p for p in input_dir.glob("*.jsonl") if p.is_file()])


def upload_files(input_files: list[Path], dry_run: bool) -> list[dict[str, str]]:
    if dry_run:
        return [
            {
                "local_file": str(path),
                "display_name": path.name,
                "mime_type": "application/jsonl",
                "name": "",
                "uri": "",
                "state": "DRY_RUN",
            }
            for path in input_files
        ]

    client = genai.Client(api_key=get_api_key())

    results: list[dict[str, str]] = []
    for path in input_files:
        uploaded = client.files.upload(
            file=path,
            config={
                "mime_type": "application/jsonl",
                "display_name": path.name,
            },
        )

        state_obj = getattr(uploaded, "state", None)
        state = get_attr(state_obj, "name", default=get_attr(uploaded, "state", ""))

        results.append(
            {
                "local_file": str(path),
                "display_name": path.name,
                "mime_type": "application/jsonl",
                "name": get_attr(uploaded, "name"),
                "uri": get_attr(uploaded, "uri"),
                "state": state,
            }
        )

    return results


def render_markdown(results: list[dict[str, str]], input_dir: Path, dry_run: bool) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# Gemini Files API Upload Records")
    lines.append("")
    lines.append(f"- generated_at: {generated_at}")
    lines.append(f"- input_dir: {input_dir}")
    lines.append(f"- mode: {'dry-run' if dry_run else 'upload'}")
    lines.append("")
    lines.append("## Uploaded Files")
    lines.append("")
    lines.append("| local_file | display_name | mime_type | file_name | file_uri | state |")
    lines.append("|---|---|---|---|---|---|")

    for item in results:
        lines.append(
            "| "
            f"{item['local_file']} | "
            f"{item['display_name']} | "
            f"{item['mime_type']} | "
            f"{item['name'] or '-'} | "
            f"{item['uri'] or '-'} | "
            f"{item['state'] or '-'} |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload JSONL files for Gemini Batch API via Files API and record metadata."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing JSONL files to upload (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--md-path",
        type=Path,
        default=DEFAULT_MD_PATH,
        help=f"Markdown output path (default: {DEFAULT_MD_PATH})",
    )
    parser.add_argument(
        "--append-md",
        action="store_true",
        help="Append to markdown file instead of overwrite.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not upload, only print/write planned file entries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    files = collect_jsonl_files(args.input_dir)
    if not files:
        print(f"No JSONL files found in: {args.input_dir}")
        return

    results = upload_files(files, dry_run=args.dry_run)

    for item in results:
        print(
            f"[done] {item['display_name']} -> "
            f"name={item['name'] or '-'} uri={item['uri'] or '-'} state={item['state'] or '-'}"
        )

    markdown = render_markdown(results, args.input_dir, args.dry_run)

    args.md_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append_md else "w"
    with args.md_path.open(mode, encoding="utf-8") as f:
        if args.append_md and args.md_path.exists() and args.md_path.stat().st_size > 0:
            f.write("\n\n")
        f.write(markdown)

    print(f"\nMarkdown written to: {args.md_path}")


if __name__ == "__main__":
    main()
