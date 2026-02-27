"""
Poll Gemini Batch jobs and download completed output files.

Reads jobs from:
  data/09_gemini_restyle_output/with_ds/batch_jobs.json

Maintains receive state in:
  data/09_gemini_restyle_output/with_ds/batch_receive_state.json

Downloads output files to:
  data/09_gemini_restyle_output/with_ds

Behavior:
- Poll every 5 minutes by default.
- Jobs already marked done in state JSON will be skipped.
- Once a job output file is downloaded, it will not be polled again.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "09_gemini_restyle_output" / "with_ds"
DEFAULT_BATCH_JOBS_JSON = DEFAULT_OUTPUT_DIR / "batch_jobs.json"
DEFAULT_RECEIVE_STATE_JSON = DEFAULT_OUTPUT_DIR / "batch_receive_state.json"
DEFAULT_POLL_SECONDS = 300

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
SUCCESS_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}

load_dotenv(PROJECT_ROOT / ".env")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY).")
    return api_key


def state_to_str(state: object) -> str:
    name = getattr(state, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(state)


def load_jobs(jobs_path: Path) -> list[dict[str, Any]]:
    if not jobs_path.exists():
        raise FileNotFoundError(f"Jobs metadata file not found: {jobs_path}")
    payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
    if not jobs:
        raise ValueError(f"No jobs found in: {jobs_path}")
    return jobs


def load_state(state_path: Path, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        payload = {"updated_at": None, "jobs": {}}

    payload.setdefault("jobs", {})
    jobs_state: dict[str, Any] = payload["jobs"]

    for job in jobs:
        batch_name = job.get("batch_name")
        if not batch_name:
            continue
        jobs_state.setdefault(
            batch_name,
            {
                "done": False,
                "received": False,
                "last_state": job.get("state", "UNKNOWN"),
                "output_files": [],
                "downloaded_paths": [],
                "last_checked_at": None,
                "error": "",
            },
        )

    return payload


def save_state(state_path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_file_names_from_obj(obj: Any, found: set[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        if re.fullmatch(r"files/[A-Za-z0-9_-]+", obj):
            found.add(obj)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            extract_file_names_from_obj(value, found)
        return
    if isinstance(obj, list):
        for item in obj:
            extract_file_names_from_obj(item, found)
        return


def extract_output_file_names(batch_job: Any, src_file_name: str) -> list[str]:
    found: set[str] = set()

    if hasattr(batch_job, "model_dump"):
        try:
            data = batch_job.model_dump(mode="json")
            extract_file_names_from_obj(data, found)
        except Exception:
            pass

    if src_file_name in found:
        found.remove(src_file_name)

    return sorted(found)


def download_file(client: genai.Client, file_name: str, output_dir: Path, batch_name: str) -> Path:
    file_id = file_name.split("/", 1)[-1]
    batch_id = batch_name.split("/", 1)[-1]

    # Try to preserve display_name when available.
    target_name = f"{batch_id}__{file_id}.jsonl"
    try:
        meta = client.files.get(name=file_name)
        display_name = getattr(meta, "display_name", None)
        if isinstance(display_name, str) and display_name:
            safe_display = re.sub(r"[^A-Za-z0-9._-]+", "_", display_name)
            target_name = f"{batch_id}__{safe_display}"
    except Exception:
        pass

    output_path = output_dir / target_name
    if output_path.exists():
        return output_path

    content = client.files.download(file=file_name)
    output_path.write_bytes(content)
    return output_path


def poll_once(
    client: genai.Client,
    jobs: list[dict[str, Any]],
    state_payload: dict[str, Any],
    output_dir: Path,
) -> tuple[int, int]:
    jobs_state: dict[str, Any] = state_payload["jobs"]

    remaining = 0
    received_now = 0

    for job in jobs:
        batch_name = job.get("batch_name")
        src_file_name = job.get("src_file_name", "")
        if not batch_name:
            continue

        state = jobs_state.setdefault(batch_name, {})
        if state.get("done"):
            continue

        remaining += 1

        try:
            batch_job = client.batches.get(name=batch_name)
            batch_state = state_to_str(getattr(batch_job, "state", "UNKNOWN"))
            state["last_state"] = batch_state
            state["last_checked_at"] = now_iso()
            state["error"] = ""

            print(f"[poll] {batch_name} -> {batch_state}")

            if batch_state in SUCCESS_STATES:
                output_files = extract_output_file_names(batch_job, src_file_name=src_file_name)
                state["output_files"] = output_files

                downloaded_paths = state.get("downloaded_paths", [])
                downloaded_set = set(downloaded_paths)

                for file_name in output_files:
                    path = download_file(client, file_name, output_dir, batch_name)
                    if str(path) not in downloaded_set:
                        downloaded_paths.append(str(path))
                        downloaded_set.add(str(path))
                        print(f"[recv] {batch_name} -> {path}")

                state["downloaded_paths"] = downloaded_paths
                if downloaded_paths:
                    state["received"] = True
                    state["done"] = True
                    received_now += 1
                else:
                    # Succeeded but no output file discovered yet; continue polling.
                    state["received"] = False
                    state["done"] = False

            elif batch_state in TERMINAL_STATES:
                # Terminal but failed/cancelled/expired; stop polling to avoid infinite loop.
                state["received"] = False
                state["done"] = True

        except Exception as exc:
            state["last_checked_at"] = now_iso()
            state["error"] = str(exc)
            print(f"[error] {batch_name}: {exc}")

    return remaining, received_now


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll Gemini Batch jobs and download outputs.")
    parser.add_argument("--jobs-json", type=Path, default=DEFAULT_BATCH_JOBS_JSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_RECEIVE_STATE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--once", action="store_true", help="Poll once then exit.")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means unlimited cycles.")
    args = parser.parse_args()

    jobs = load_jobs(args.jobs_json)
    state_payload = load_state(args.state_json, jobs)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=get_api_key())

    cycle = 0
    while True:
        cycle += 1
        print(f"\n=== Poll cycle {cycle} @ {now_iso()} ===")

        remaining, received_now = poll_once(
            client=client,
            jobs=jobs,
            state_payload=state_payload,
            output_dir=args.output_dir,
        )
        save_state(args.state_json, state_payload)

        done_count = sum(1 for value in state_payload["jobs"].values() if value.get("done"))
        total = len([job for job in jobs if job.get("batch_name")])

        print(
            f"[summary] total={total} done={done_count} still_polling={total - done_count} received_now={received_now}"
        )
        print(f"[state] {args.state_json}")

        if args.once:
            break

        if done_count >= total:
            print("All jobs are done. Stop polling.")
            break

        if args.max_cycles and cycle >= args.max_cycles:
            print(f"Reached max cycles: {args.max_cycles}")
            break

        print(f"Sleep {args.poll_seconds} seconds before next poll...")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
