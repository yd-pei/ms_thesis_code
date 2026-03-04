#!/usr/bin/env bash
set -euo pipefail

# Run raw_judge (forward-pass, bf16) on restyled pairwise files in 095.
#
# The PRIMARY model (before "__" in the filename) is used as the judge.
# Only runs normal mode (no --swap-answers), since we already have
# position-independent samples and only need to re-evaluate after restyle.
#
# Usage:
#   bash scripts/run_raw_judge_095.sh
#   bash scripts/run_raw_judge_095.sh --input-dir data/095_restyled_pairwise --output-dir data/065_position_independent/after_restyle

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Load .env if present (for HF_TOKEN etc.) ─────────────────────────
ENV_FILE="$ROOT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

INPUT_DIR="$ROOT_DIR/data/095_restyled_pairwise"
QUALITY_PATH="$ROOT_DIR/data/01_processed_quality/quality_train.jsonl"
OUTPUT_DIR="$ROOT_DIR/data/065_position_independent/after_restyle"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_raw_judge_095.sh [options]

Options:
  --input-dir <path>     Input directory (default: data/095_restyled_pairwise)
  --quality-path <path>  Quality file path (default: data/01_processed_quality/quality_train.jsonl)
  --output-dir <path>    Output directory (default: data/065_position_independent/after_restyle)
  -h, --help             Show this help message

For each restyled pairwise file {primary}__{opponent}__raw_judge.jsonl,
the primary model is used as the judge. Only runs normal mode (no swap).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"
      [[ "$INPUT_DIR" != /* ]] && INPUT_DIR="$ROOT_DIR/$INPUT_DIR"
      shift 2
      ;;
    --quality-path)
      QUALITY_PATH="$2"
      [[ "$QUALITY_PATH" != /* ]] && QUALITY_PATH="$ROOT_DIR/$QUALITY_PATH"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      [[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
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

# ── Model name mapping ────────────────────────────────────────────────
declare -A MODEL_MAP=(
  ["llama31_70b"]="meta-llama/Llama-3.1-70B-Instruct"
  ["llama31_8b"]="meta-llama/Llama-3.1-8B-Instruct"
  ["llama33_70b"]="meta-llama/Llama-3.3-70B-Instruct"
  ["qwen25_7b"]="Qwen/Qwen2.5-7B-Instruct"
)

# ── Validation ────────────────────────────────────────────────────────
command -v uv >/dev/null 2>&1 || { echo "[error] uv not found in PATH."; exit 1; }
[[ -d "$INPUT_DIR" ]] || { echo "[error] Input dir not found: $INPUT_DIR"; exit 1; }
[[ -f "$QUALITY_PATH" ]] || { echo "[error] Quality file not found: $QUALITY_PATH"; exit 1; }
mkdir -p "$OUTPUT_DIR"

# ── Discover and process files ────────────────────────────────────────
run_count=0
skip_count=0

for data_path in "$INPUT_DIR"/*.jsonl; do
  [[ -f "$data_path" ]] || continue

  file_name="$(basename "$data_path")"
  base_name="${file_name%.jsonl}"

  # Extract primary model name (before first "__")
  primary="${base_name%%__*}"

  # Look up HuggingFace model path
  hf_model="${MODEL_MAP[$primary]:-}"
  if [[ -z "$hf_model" ]]; then
    echo "[skip] Unknown primary model '$primary' in $file_name (no HF mapping)"
    skip_count=$((skip_count + 1))
    continue
  fi

  # Output keeps same filename (e.g. llama33_70b__ds_v3__raw_judge.jsonl)
  output_path="$OUTPUT_DIR/$file_name"

  echo ""
  echo "========================================"
  echo "[run] file=$file_name judge=$primary ($hf_model) mode=normal"
  echo "========================================"
  uv run align_lab raw_judge \
    --model "$hf_model" \
    --data-path "$data_path" \
    --quality-path "$QUALITY_PATH" \
    --output-path "$output_path" \
    --batch-size 2
  run_count=$((run_count + 1))

done

echo ""
echo "[done] Completed $run_count runs ($skip_count skipped). Outputs in: $OUTPUT_DIR"
