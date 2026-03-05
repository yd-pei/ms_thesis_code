#!/usr/bin/env bash
set -euo pipefail

# Run judge/raw_judge on official QuALITY pairwise inputs under data/official.
#
# Input files:
#   data/official/{before_restyle,after_restyle}/*.jsonl
#
# Context file:
#   data/official/quality_official_context.jsonl
#
# Output directories (by default):
#   data/official/judged_before
#   data/official/judged_after
#
# Usage examples:
#   bash scripts/run_official_judge.sh
#   bash scripts/run_official_judge.sh --mode judge --phase before
#   bash scripts/run_official_judge.sh --mode both --phase both --with-swap
#   bash scripts/run_official_judge.sh --dry-run

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env if present (HF_TOKEN etc.)
ENV_FILE="$ROOT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

MODE="raw_judge"            # raw_judge | judge | both
PHASE="both"                # before | after | both
INPUT_ROOT="$ROOT_DIR/data/official"
QUALITY_PATH="$ROOT_DIR/data/official/quality_official_context.jsonl"
OUTPUT_ROOT="$ROOT_DIR/data/official"
RAW_BATCH_SIZE=2
RAW_MAX_BATCH_TOKENS=0
JUDGE_CONFIG="$ROOT_DIR/configs/judge.yaml"
JUDGE_QUANTIZATION="bitsandbytes"
WITH_SWAP=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_official_judge.sh [options]

Options:
  --mode <raw_judge|judge|both>      Inference mode (default: raw_judge)
  --phase <before|after|both>        Dataset phase (default: both)
  --input-root <path>                Root containing before_restyle/after_restyle (default: data/official)
  --quality-path <path>              Context JSONL (default: data/official/quality_official_context.jsonl)
  --output-root <path>               Root for judged outputs (default: data/official)
  --raw-batch-size <int>             raw_judge batch size (default: 2)
  --raw-max-batch-tokens <int>       raw_judge token budget per batch after padding (default: 0=disabled)
  --judge-config <path>              judge config YAML (default: configs/judge.yaml)
  --judge-quantization <q>           judge quantization: bitsandbytes|gptq|awq|fp8|none (default: bitsandbytes)
  --with-swap                        Also run swapped answers
  --dry-run                          Print commands without executing
  -h, --help                         Show this help message

Outputs:
  before -> <output-root>/judged_before/*.jsonl
  after  -> <output-root>/judged_after/*.jsonl
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --phase)
      PHASE="$2"
      shift 2
      ;;
    --input-root)
      INPUT_ROOT="$2"
      [[ "$INPUT_ROOT" != /* ]] && INPUT_ROOT="$ROOT_DIR/$INPUT_ROOT"
      shift 2
      ;;
    --quality-path)
      QUALITY_PATH="$2"
      [[ "$QUALITY_PATH" != /* ]] && QUALITY_PATH="$ROOT_DIR/$QUALITY_PATH"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      [[ "$OUTPUT_ROOT" != /* ]] && OUTPUT_ROOT="$ROOT_DIR/$OUTPUT_ROOT"
      shift 2
      ;;
    --raw-batch-size)
      RAW_BATCH_SIZE="$2"
      shift 2
      ;;
    --raw-max-batch-tokens)
      RAW_MAX_BATCH_TOKENS="$2"
      shift 2
      ;;
    --judge-config)
      JUDGE_CONFIG="$2"
      [[ "$JUDGE_CONFIG" != /* ]] && JUDGE_CONFIG="$ROOT_DIR/$JUDGE_CONFIG"
      shift 2
      ;;
    --judge-quantization)
      JUDGE_QUANTIZATION="$2"
      shift 2
      ;;
    --with-swap)
      WITH_SWAP=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ "$MODE" != "raw_judge" && "$MODE" != "judge" && "$MODE" != "both" ]]; then
  echo "[error] --mode must be one of: raw_judge, judge, both"
  exit 1
fi
if [[ "$PHASE" != "before" && "$PHASE" != "after" && "$PHASE" != "both" ]]; then
  echo "[error] --phase must be one of: before, after, both"
  exit 1
fi
if [[ "$JUDGE_QUANTIZATION" != "bitsandbytes" && "$JUDGE_QUANTIZATION" != "gptq" && "$JUDGE_QUANTIZATION" != "awq" && "$JUDGE_QUANTIZATION" != "fp8" && "$JUDGE_QUANTIZATION" != "none" ]]; then
  echo "[error] --judge-quantization must be one of: bitsandbytes, gptq, awq, fp8, none"
  exit 1
fi

model_from_alias() {
  local alias="$1"
  case "$alias" in
    llama31_8b) echo "meta-llama/Llama-3.1-8B-Instruct" ;;
    qwen25_7b) echo "Qwen/Qwen2.5-7B-Instruct" ;;
    *) echo "" ;;
  esac
}

command -v uv >/dev/null 2>&1 || { echo "[error] uv not found in PATH."; exit 1; }
[[ -d "$INPUT_ROOT" ]] || { echo "[error] Input root not found: $INPUT_ROOT"; exit 1; }
[[ -f "$QUALITY_PATH" ]] || { echo "[error] Quality file not found: $QUALITY_PATH"; exit 1; }
if [[ "$MODE" == "judge" || "$MODE" == "both" ]]; then
  [[ -f "$JUDGE_CONFIG" ]] || { echo "[error] Judge config not found: $JUDGE_CONFIG"; exit 1; }
fi

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

phase_to_input_dir() {
  local phase="$1"
  if [[ "$phase" == "before" ]]; then
    echo "$INPUT_ROOT/before_restyle"
  else
    echo "$INPUT_ROOT/after_restyle"
  fi
}

phase_to_output_dir() {
  local phase="$1"
  if [[ "$phase" == "before" ]]; then
    echo "$OUTPUT_ROOT/judged_before"
  else
    echo "$OUTPUT_ROOT/judged_after"
  fi
}

declare -a phases
if [[ "$PHASE" == "both" ]]; then
  phases=("before" "after")
else
  phases=("$PHASE")
fi

run_count=0
skip_count=0

for phase in "${phases[@]}"; do
  input_dir="$(phase_to_input_dir "$phase")"
  output_dir="$(phase_to_output_dir "$phase")"
  [[ -d "$input_dir" ]] || { echo "[warn] Missing phase dir: $input_dir (skip)"; continue; }
  mkdir -p "$output_dir"

  for data_path in "$input_dir"/*.jsonl; do
    [[ -f "$data_path" ]] || continue

    file_name="$(basename "$data_path")"
    base_name="${file_name%.jsonl}"
    primary="${base_name%%__*}"

    hf_model="$(model_from_alias "$primary")"
    if [[ -z "$hf_model" ]]; then
      echo "[skip] phase=$phase file=$file_name unknown judge alias '$primary'"
      skip_count=$((skip_count + 1))
      continue
    fi

    if [[ "$MODE" == "raw_judge" || "$MODE" == "both" ]]; then
      output_normal="$output_dir/${base_name}__raw_judge.jsonl"
      echo ""
      echo "========================================"
      echo "[run] phase=$phase mode=raw_judge file=$file_name judge=$primary ($hf_model)"
      echo "========================================"
      run_cmd uv run align_lab raw_judge \
        --model "$hf_model" \
        --data-path "$data_path" \
        --quality-path "$QUALITY_PATH" \
        --output-path "$output_normal" \
        --batch-size "$RAW_BATCH_SIZE" \
        --max-batch-tokens "$RAW_MAX_BATCH_TOKENS"
      run_count=$((run_count + 1))

      if [[ "$WITH_SWAP" -eq 1 ]]; then
        output_swapped="$output_dir/${base_name}__raw_judge_swapped.jsonl"
        echo ""
        echo "========================================"
        echo "[run] phase=$phase mode=raw_judge_swap file=$file_name judge=$primary ($hf_model)"
        echo "========================================"
        run_cmd uv run align_lab raw_judge \
          --model "$hf_model" \
          --data-path "$data_path" \
          --quality-path "$QUALITY_PATH" \
          --output-path "$output_swapped" \
          --swap-answers \
          --batch-size "$RAW_BATCH_SIZE" \
          --max-batch-tokens "$RAW_MAX_BATCH_TOKENS"
        run_count=$((run_count + 1))
      fi
    fi

    if [[ "$MODE" == "judge" || "$MODE" == "both" ]]; then
      output_normal="$output_dir/${base_name}__judge.jsonl"
      echo ""
      echo "========================================"
      echo "[run] phase=$phase mode=judge file=$file_name judge=$primary ($hf_model)"
      echo "========================================"
      run_cmd uv run align_lab judge \
        --model "$hf_model" \
        --quantization "$JUDGE_QUANTIZATION" \
        --data-path "$data_path" \
        --quality-path "$QUALITY_PATH" \
        --output-path "$output_normal" \
        --config-path "$JUDGE_CONFIG"
      run_count=$((run_count + 1))

      if [[ "$WITH_SWAP" -eq 1 ]]; then
        output_swapped="$output_dir/${base_name}__judge_swapped.jsonl"
        echo ""
        echo "========================================"
        echo "[run] phase=$phase mode=judge_swap file=$file_name judge=$primary ($hf_model)"
        echo "========================================"
        run_cmd uv run align_lab judge \
          --model "$hf_model" \
          --quantization "$JUDGE_QUANTIZATION" \
          --data-path "$data_path" \
          --quality-path "$QUALITY_PATH" \
          --output-path "$output_swapped" \
          --swap-answers \
          --config-path "$JUDGE_CONFIG"
        run_count=$((run_count + 1))
      fi
    fi
  done
done

echo ""
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[done] Dry run finished. Planned $run_count runs ($skip_count skipped)."
else
  echo "[done] Completed $run_count runs ($skip_count skipped). Outputs under: $OUTPUT_ROOT"
fi
