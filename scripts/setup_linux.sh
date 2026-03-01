#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HF_REPO="yidingp/mitigate_preference_dpo"
LOCAL_DIR="./data"
REPO_TYPE="dataset"
HF_TOKEN="${HF_TOKEN:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_linux.sh [options]

Options:
  --hf-token <token>      Hugging Face token (highest priority)
  --repo <repo_id>        HF repo id (default: yidingp/mitigate_preference_dpo)
  --local-dir <path>      Download directory (default: ./data)
  --repo-type <type>      HF repo type (default: dataset)
  -h, --help              Show this help message

Examples:
  bash scripts/setup_linux.sh
  bash scripts/setup_linux.sh --hf-token hf_xxx
  HF_TOKEN=hf_xxx bash scripts/setup_linux.sh

Token priority:
  1) --hf-token
  2) HF_TOKEN environment variable
  3) .env file (HF_TOKEN or HUGGINGFACE_HUB_TOKEN)
EOF
}

load_hf_token_from_env_file() {
  local env_file="$ROOT_DIR/.env"
  if [[ ! -f "$env_file" ]]; then
    return 0
  fi

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    local line="$raw_line"
    line="${line#${line%%[![:space:]]*}}"
    line="${line%${line##*[![:space:]]}}"

    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    [[ "$line" != *=* ]] && continue

    local key="${line%%=*}"
    local value="${line#*=}"

    key="${key#${key%%[![:space:]]*}}"
    key="${key%${key##*[![:space:]]}}"
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"

    if [[ "$value" =~ ^\".*\"$ ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" =~ ^\'.*\'$ ]]; then
      value="${value:1:${#value}-2}"
    fi

    case "$key" in
      HF_TOKEN|HUGGINGFACE_HUB_TOKEN)
        if [[ -z "$HF_TOKEN" && -n "$value" ]]; then
          HF_TOKEN="$value"
          return 0
        fi
        ;;
    esac
  done < "$env_file"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --repo)
      HF_REPO="$2"
      shift 2
      ;;
    --local-dir)
      LOCAL_DIR="$2"
      shift 2
      ;;
    --repo-type)
      REPO_TYPE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

load_hf_token_from_env_file

cd "$ROOT_DIR"

echo "[1/4] Upgrading pip..."
if ! python3 -m pip install --upgrade pip; then
  echo "[warn] System-level pip upgrade failed; retrying with --user"
  python3 -m pip install --user --upgrade pip
fi

echo "[2/4] Installing/upgrading uv..."
if ! python3 -m pip install --upgrade uv; then
  echo "[warn] System-level uv install failed; retrying with --user"
  python3 -m pip install --user --upgrade uv
fi

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || {
  echo "[error] uv is not found in PATH after installation."
  echo "        Try: export PATH=\"$HOME/.local/bin:\$PATH\""
  exit 1
}

echo "[info] uv version: $(uv --version)"

echo "[3/4] Syncing project dependencies with uv..."
uv sync

echo "[4/4] Downloading dataset from Hugging Face..."
mkdir -p "$LOCAL_DIR"

if [[ -n "$HF_TOKEN" ]]; then
  uv run hf download "$HF_REPO" --local-dir "$LOCAL_DIR" --repo-type "$REPO_TYPE" --token "$HF_TOKEN"
else
  uv run hf download "$HF_REPO" --local-dir "$LOCAL_DIR" --repo-type "$REPO_TYPE"
fi

echo "[done] Environment setup completed."
echo "       Repo: $HF_REPO"
echo "       Local dir: $LOCAL_DIR"
