# Preference Mitigation

## Run

### Install

```bash
uv sync
uv run hf download yidingp/mitigate_preference_dpo --local-dir ./data --repo-type dataset
```

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

`huggingface-cli login` is optional for this project (not required if `HF_TOKEN`/`--hf-token` is provided).

### Inference

```bash
uv run align_lab inference \
    --model gpt-3.5-turbo \
    --backend openai \
    --api-key "sk-..."
```

```bash
uv run align_lab inference \
    --model claude-3-5-sonnet-20240620 \
    --backend anthropic \
    --api-key "sk-ant-..."
```

```bash
uv run align_lab inference \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend vllm \
    --hf-token "<hf_token>"
```

### Judge (pairwise inference)

```bash
uv run align_lab judge \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --data-path data/04_clean_pairwise_output/llama31_70b__ds_v3.jsonl \
    --quality-path data/01_processed_quality/quality_train.jsonl \
    --hf-token "<hf_token>" \
    --output-path outputs/judge_results.jsonl
```

```bash
uv run align_lab judge \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --data-path data/04_clean_pairwise_output/llama31_70b__ds_v3.jsonl \
    --quality-path data/01_processed_quality/quality_train.jsonl \
    --hf-token "<hf_token>" \
    --output-path outputs/judge_results_swapped.jsonl \
    --swap-answers
```
