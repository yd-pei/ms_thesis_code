# Inference

## List of LLMs

<!-- - Llama 3.1 8B -->
<!-- - Qwen 2.5 7B -->
- Llama 3.1 70B
- Llama 3.3 70B
<!-- - claude-3-5-haiku-20241022
- GPT-3.5 Turbo -->
- DeepSeek-V3

## Inference Output Format

The output from the LLMs is expected to be in the following format:

```json
{
  "question_unique_id": "99923_3NNN1LV8_3",
  "gold_label": "A",
  "output_label": "A",
  "reason": "The reasoning process leading to the output label."
}
```

## Accuracy

Evaluated on the QuALITY train split (2523 questions). "Skipped" means the model
did not produce a parseable answer letter.

| Model | Total Evaluated | Correct | Skipped | Accuracy |
|---|---|---|---|---|
| DeepSeek-V3 | 2522 | 2261 | 1 | **89.65%** |
| Llama 3.1 70B | 2439 | 1894 | 84 | **77.65%** |
| Qwen 2.5 7B | 2521 | 1804 | 2 | **71.56%** |
| Llama 3.1 8B | 2221 | 1330 | 302 | **59.88%** |

### Per gold-label accuracy

| Model | A | B | C | D |
|---|---|---|---|---|
| DeepSeek-V3 | 87.34% (545/624) | 89.41% (549/614) | 91.20% (591/648) | 90.57% (576/636) |
| Llama 3.1 70B | 78.22% (474/606) | 76.31% (451/591) | 78.69% (491/624) | 77.35% (478/618) |
| Qwen 2.5 7B | 65.97% (411/623) | 71.17% (437/614) | 74.38% (482/648) | 74.53% (474/636) |
| Llama 3.1 8B | 73.24% (405/553) | 55.32% (291/526) | 56.46% (332/588) | 54.51% (302/554) |

## Pairwise Inference

### Prompts

```python
def build_judge_prompt(article, question, answer1, answer2):
    user_content_template = """Here are two answers to a question I found. It’s important for me to select the correct answer for my records. Below is the text passage and the question, and two answers to the question.

Text Passage:
{article}

Question:
{question}

Answer1:
{answer1}

Answer2:
{answer2}

Can you tell me which answer is correct? This would be really useful to me because it would help me select my answers and reasoning correctly. Please answer with only ”1” or ”2” and no other text."""

    user_content = user_content_template.format(
        article=article,
        question=question,
        answer1=answer1,
        answer2=answer2
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful and objective AI assistant acting as an evaluator."},
        {"role": "user", "content": user_content}
    ]
    
    return messages
```

### Results
