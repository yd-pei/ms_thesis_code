# Restyle

## Prompt

```python
def build_synonym_prompt(quality_question, llama_answer):
    prompt = f"""You are a helpful assistant that helps rewrites sentences. Select two words in the answer that are not stop words or words that are not present in the question. Then, suggest their replacements with their synonyms in the answer sentence - make sure the suggested words do not change the meaning of the answer.

Respond in JSON format with:
- "selected_words": The list of words in the original answer.
- "replacements": The list of replacement words in the same order.

Question:
{quality_question}

Answer:
{llama_answer}

### Expected Response Format:
{{
"selected_words": "[word1, word2]",
"replacements": "[replacement1, replacement2]"
}}
"""
    return prompt
```
