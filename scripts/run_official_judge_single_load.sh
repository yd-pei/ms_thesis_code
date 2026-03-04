#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${UV_CACHE_DIR:-}" ]]; then
  export UV_CACHE_DIR=/tmp/uv-cache
fi

exec uv run python "$ROOT_DIR/scripts/run_official_judge_single_load.py" "$@"
