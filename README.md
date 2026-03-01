# Preference Mitigation

## Run

### Install

```bash
uv sync
uv run hf download yidingp/mitigate_preference_dpo --local-dir ./data --repo-type dataset
```

Note: `vllm` and `bitsandbytes` are configured as non-Darwin dependencies (`platform_system != 'Darwin'`). On macOS, `uv sync` will skip them automatically. Use API backends (`openai` / `anthropic`) on macOS, or run local `vllm` inference in a non-Darwin environment.

### HuggingFace Token

Copy the sample env file and fill in your token:

```bash
cp .env.example .env
# Then edit .env and replace hf_your_token_here with your actual token
```

The token will be automatically loaded via `.env`. You can also pass it directly in the command with `--hf-token "hf_xxx"`.

It is needed for local model loading (inference/judge/raw_judge), not for OpenAI/Anthropic API calls.

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

### Raw Judge (judge with transformers)

```bash
uv run align_lab raw_judge \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --data-path data/04_clean_pairwise_output/llama31_70b__ds_v3.jsonl \
    --quality-path data/01_processed_quality/quality_train.jsonl \
    --output-path outputs/llama_31_70_judge_results.jsonl \
    --swap-answers
```

### Restyle

```bash
uv run python scripts/restyle/poll_and_download_gemini_batches.py
```

```bash
uv run align_lab judge \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --data-path data/11_restyled_position_independent_response/ds/llama33_70b_judge.jsonl \
    --quality-path data/01_processed_quality/quality_train.jsonl \
    --config-path configs/judge.yaml \
    --hf-token "<hf_token>" \
    --output-path outputs/llama33_70b.jsonl
```
