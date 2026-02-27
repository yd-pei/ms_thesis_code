# Preference Mitigation

## Run

### Install

```bash
uv sync
uv run hf download yidingp/mitigate_preference_dpo --local-dir ./data --repo-type dataset
```

Note: `vllm` and `bitsandbytes` are configured as non-Darwin dependencies (`platform_system != 'Darwin'`). On macOS, `uv sync` will skip them automatically. Use API backends (`openai` / `anthropic`) on macOS, or run local `vllm` inference in a non-Darwin environment.

### HuggingFace Token

For gated/private models, you can use either method:

```bash
# Option 1: environment variable (recommended)
export HF_TOKEN="hf_xxx"
```

```bash
# Option 2: pass token directly in command
--hf-token "hf_xxx"
```

`--hf-token` can be placed anywhere in the command options (usually after `--model` for readability).
It is needed for local `vllm` model loading (inference/judge), not for OpenAI/Anthropic API calls.

`huggingface-cli login` is optional for this project (not required if `HF_TOKEN`/`--hf-token` is provided).

### Inference

```bash
uv run align_lab inference \
    --model gpt-3.5-turbo \
    --backend openai \
    --config-path configs/inference.yaml \
    --api-key "sk-..."
```

```bash
uv run align_lab inference \
    --model claude-3-5-sonnet-20240620 \
    --backend anthropic \
    --config-path configs/inference.yaml \
    --api-key "sk-ant-..."
```

```bash
uv run align_lab inference \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend vllm \
    --config-path configs/inference.yaml \
    --hf-token "<hf_token>"
```

### Judge (pairwise inference)

```bash
uv run align_lab judge \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --data-path data/04_clean_pairwise_output/llama31_70b__ds_v3.jsonl \
    --quality-path data/01_processed_quality/quality_train.jsonl \
    --config-path configs/judge.yaml \
    --hf-token "<hf_token>" \
    --output-path outputs/judge_results.jsonl
```

```bash
uv run align_lab judge \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --data-path data/04_clean_pairwise_output/llama31_70b__ds_v3.jsonl \
    --quality-path data/01_processed_quality/quality_train.jsonl \
    --config-path configs/judge.yaml \
    --hf-token "<hf_token>" \
    --output-path outputs/judge_results_swapped.jsonl \
    --swap-answers
```

### Restyle

```bash
uv run python scripts/restyle/poll_and_download_gemini_batches.py
```
