#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="$ROOT_DIR/data/11_restyled_response/raw"
QUALITY_PATH="$ROOT_DIR/data/01_processed_quality/quality_train.jsonl"
OUTPUT_DIR="$ROOT_DIR/outputs"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_raw_judge_restyled.sh [options]

Options:
  --input-dir <path>     Input directory (default: data/11_restyled_response/raw)
  --quality-path <path>  Quality file path (default: data/01_processed_quality/quality_train.jsonl)
  --output-dir <path>    Output directory (default: outputs)
  -h, --help             Show this help message

This script runs raw_judge for 4 models, each with:
  1) normal mode
  2) --swap-answers mode
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --quality-path)
      QUALITY_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
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

if [[ "$INPUT_DIR" != /* ]]; then
  INPUT_DIR="$ROOT_DIR/$INPUT_DIR"
fi
if [[ "$QUALITY_PATH" != /* ]]; then
  QUALITY_PATH="$ROOT_DIR/$QUALITY_PATH"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi

command -v uv >/dev/null 2>&1 || {
  echo "[error] uv not found in PATH."
  exit 1
}

[[ -d "$INPUT_DIR" ]] || { echo "[error] Input dir not found: $INPUT_DIR"; exit 1; }
[[ -f "$QUALITY_PATH" ]] || { echo "[error] Quality file not found: $QUALITY_PATH"; exit 1; }
mkdir -p "$OUTPUT_DIR"

FILES=(
  "llama31_70b_judge.jsonl"
  "llama31_8b_judge.jsonl"
  "llama33_70b_judge.jsonl"
  "qwen25_7b_judge.jsonl"
)

MODELS=(
  "meta-llama/Llama-3.1-70B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

run_count=0
for i in "${!FILES[@]}"; do
  file_name="${FILES[$i]}"
  model_name="${MODELS[$i]}"
  data_path="$INPUT_DIR/$file_name"

  [[ -f "$data_path" ]] || { echo "[error] Missing input file: $data_path"; exit 1; }

  base_name="${file_name%.jsonl}"
  output_normal="$OUTPUT_DIR/${base_name}__raw_judge.jsonl"
  output_swapped="$OUTPUT_DIR/${base_name}__raw_judge_swapped.jsonl"

  echo "[run] model=$model_name file=$file_name mode=normal"
  uv run align_lab raw_judge \
    --model "$model_name" \
    --data-path "$data_path" \
    --quality-path "$QUALITY_PATH" \
    --output-path "$output_normal"
  run_count=$((run_count + 1))

  echo "[run] model=$model_name file=$file_name mode=swap"
  uv run align_lab raw_judge \
    --model "$model_name" \
    --data-path "$data_path" \
    --quality-path "$QUALITY_PATH" \
    --output-path "$output_swapped" \
    --swap-answers
  run_count=$((run_count + 1))
done

echo "[done] Completed $run_count runs. Outputs in: $OUTPUT_DIR"
