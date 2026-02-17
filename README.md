# Preference Mitigation

## Run

### Install

```bash
uv sync
uv run hf download yidingp/mitigate_preference_dpo --local-dir ./data --repo-type dataset
```

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
