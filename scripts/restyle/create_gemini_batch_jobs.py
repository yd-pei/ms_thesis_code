"""
Create Gemini Batch jobs from uploaded file records.

Input markdown (default):
  doc/restyle/gemini_file_api.md

Output metadata files:
  data/09_gemini_restyle_output/with_ds/batch_jobs.json
  data/09_gemini_restyle_output/with_ds/batch_receive_state.json

This script only creates batch jobs and records metadata. It does not poll or download outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_UPLOAD_MD = PROJECT_ROOT / "doc" / "restyle" / "gemini_file_api.md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "09_gemini_restyle_output" / "with_ds"
DEFAULT_BATCH_JOBS_JSON = DEFAULT_OUTPUT_DIR / "batch_jobs.json"
DEFAULT_RECEIVE_STATE_JSON = DEFAULT_OUTPUT_DIR / "batch_receive_state.json"

load_dotenv(PROJECT_ROOT / ".env")


def get_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY).")
    return api_key


def parse_upload_markdown(md_path: Path) -> list[dict[str, str]]:
    if not md_path.exists():
        raise FileNotFoundError(f"Upload markdown not found: {md_path}")

    lines = md_path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, str]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if "---" in stripped:
            continue

        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) != 6:
            continue

        if parts[0] == "local_file":
            continue

        local_file, display_name, mime_type, file_name, file_uri, state = parts
        if not file_name.startswith("files/"):
            continue

        records.append(
            {
                "local_file": local_file,
                "display_name": display_name,
                "mime_type": mime_type,
                "file_name": file_name,
                "file_uri": file_uri,
                "state": state,
            }
        )

    if not records:
        raise ValueError(f"No valid uploaded file rows found in: {md_path}")

    return records


def state_to_str(state: object) -> str:
    name = getattr(state, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(state)


def load_existing_jobs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("jobs", [])
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Gemini Batch jobs from uploaded files.")
    parser.add_argument("--upload-md", type=Path, default=DEFAULT_UPLOAD_MD)
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--display-prefix", type=str, default="restyle-with-ds")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Create jobs even if same file already exists in batch_jobs.json")
    args = parser.parse_args()

    uploaded_files = parse_upload_markdown(args.upload_md)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_json_path = output_dir / "batch_jobs.json"
    receive_state_path = output_dir / "batch_receive_state.json"

    existing_jobs = load_existing_jobs(jobs_json_path)
    existing_by_file = {item.get("src_file_name"): item for item in existing_jobs if item.get("src_file_name")}

    client = genai.Client(api_key=get_api_key())
    created_jobs: list[dict] = []

    for item in uploaded_files:
        src_file_name = item["file_name"]

        if not args.force and src_file_name in existing_by_file:
            print(f"[skip] already created for {src_file_name}: {existing_by_file[src_file_name].get('batch_name')}")
            continue

        display_name = f"{args.display_prefix}-{item['display_name'].replace('.jsonl', '')}"

        job = client.batches.create(
            model=args.model,
            src=src_file_name,
            config={"display_name": display_name},
        )

        job_record = {
            "batch_name": job.name,
            "display_name": display_name,
            "model": args.model,
            "src_file_name": src_file_name,
            "src_local_file": item.get("local_file", ""),
            "src_display_name": item.get("display_name", ""),
            "state": state_to_str(getattr(job, "state", "UNKNOWN")),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        created_jobs.append(job_record)

        print(f"[done] {src_file_name} -> {job.name}")

    merged_jobs = existing_jobs + created_jobs
    jobs_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "upload_md": str(args.upload_md),
        "jobs": merged_jobs,
    }
    jobs_json_path.write_text(json.dumps(jobs_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if receive_state_path.exists():
        state_payload = json.loads(receive_state_path.read_text(encoding="utf-8"))
    else:
        state_payload = {"updated_at": None, "jobs": {}}

    state_jobs = state_payload.setdefault("jobs", {})
    for job in created_jobs:
        if job["batch_name"] not in state_jobs:
            state_jobs[job["batch_name"]] = {
                "received": False,
                "last_state": job["state"],
                "output_files": [],
                "downloaded_paths": [],
                "last_checked_at": None,
            }

    state_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    receive_state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nJobs metadata: {jobs_json_path}")
    print(f"Receive state:  {receive_state_path}")
    print(f"Created jobs:   {len(created_jobs)}")


if __name__ == "__main__":
    main()
