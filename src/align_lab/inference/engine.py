
import os
import json
import time
from typing import List, Optional
import re
from tqdm import tqdm

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


ONE_SHOT_TEMPLATE = """Read the following articles and answer the multiple-choice questions based on the text.

==================================================
Article:
The sun was setting behind the mountains, casting a golden hue over the valley. Sarah packed her hiking gear, making sure she had her flashlight. She knew the descent would be dark.

Question: Why did Sarah pack a flashlight?
(A) To signal for help
(B) Because she expected it to get dark during her descent
(C) To explore a cave
(D) Because it was broken

Reasoning: The text explicitly states that the sun was setting and Sarah "knew the descent would be dark." Therefore, she packed the flashlight to see in the dark. This aligns with option (B).
Answer: (B)
==================================================
Article:
{article_text}

Question: {question_text}
{options_formatted}

Reasoning:"""

def run_inference(model_path):
    pass

def format_quality_prompt(article: str, question: str, options: List[str]) -> str:
    """
    Format the input for a base LLM using the one-shot template.
    """
    options_formatted = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
    return ONE_SHOT_TEMPLATE.format(
        article_text=article,
        question_text=question,
        options_formatted=options_formatted
    )


def extract_answer_components(output_text: str):
    """
    Extracts the reasoning and the answer label from the model output.
    Expected format: " [Reasoning text...] Answer: (X)" or similar variations.
    """
    # 1. Extract Answer Label
    # Look for patterns like "Answer: (A)", "Answer: A", or just "(A)" at the end
    label_match = re.search(r"Answer:\s*\(?([A-Da-d])\)?", output_text, re.IGNORECASE)
    if not label_match:
        # Fallback: look for just (A)/(B)/(C)/(D) at the very end of the string
        label_match = re.search(r"\(?([A-Da-d])\)?\s*$", output_text)

    output_label = label_match.group(1).upper() if label_match else None

    # 2. Extract Reasoning
    # Assuming reasoning comes before "Answer:"
    if label_match:
        # Everything up to the Answer label part
        reasoning = output_text[:label_match.start()].strip()
        # Clean up "Reasoning:" prefix if it exists
        reasoning = re.sub(r"^Reasoning:\s*", "", reasoning, flags=re.IGNORECASE).strip()
    else:
        # If no label found, treat the whole thing as potentially reasoning (or malformed)
        reasoning = output_text.strip()
        
    return output_label, reasoning

def run_offline_inference(model_path: str, data_path: str, output_path: str, hf_token: str = None):
    """
    Run offline inference using vLLM on QuALITY dataset.
    """
    try:
        from vllm import LLM, SamplingParams
        import torch
    except ImportError:
        raise ImportError(
            "vLLM is not installed or not supported on this device. "
            "To use local inference, please install vllm (Linux/CUDA only). "
            "For Mac users, please use --backend openai/anthropic/deepseek."
        )

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # 1. Load Data
    print(f"Loading data from {data_path}...")
    prompts = []
    records = []
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompt = format_quality_prompt(
                data["article"], 
                data["question"], 
                data["options"]
            )
            prompts.append(prompt)
            
            # Convert gold_label from 1234 to ABCD
            if "gold_label" in data:
                try:
                    # Assuming gold_label is 1-based index (1, 2, 3, 4)
                    # or potentially string "1", "2", ...
                    idx = int(data["gold_label"]) - 1
                    if 0 <= idx < 26:
                        data["gold_label"] = chr(65 + idx)
                except (ValueError, TypeError):
                    pass # Keep as is if conversion fails

            records.append(data)

    # 2. Initialize vLLM
    print(f"Initializing vLLM with model: {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype='bfloat16',
        quantization='fp8',
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager = True
    )
    
    # 3. Define Sampling Params (Greedy decoding for deterministic answers, or slight temp)
    # Using specific stop tokens to prevent run-on generation
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=64,
        stop=["\nArticle:", "\nQuestion:", "=================================================="]
    )

    # 4. Generate
    print(f"Generating responses for {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. Save Results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving results to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record, output in zip(records, outputs):
            generated_text = output.outputs[0].text.strip()
            
            output_label, reason = extract_answer_components(generated_text)
            
            # Construct the final output record
            final_record = {
                "question_unique_id": record.get("question_unique_id"),
                "article": record.get("article"),
                "gold_label": record.get("gold_label"),
                "output_label": output_label,
                "reason": reason,
                "raw_generation": generated_text # keeping raw output for debugging
            }
            f.write(json.dumps(final_record) + "\n")
    
    print("Inference completed.")


def format_quality_chat_messages(article: str, question: str, options: List[str]) -> List[dict]:
    """
    Format the input for chat-based LLMs (OpenAI/Claude/DeepSeek) using messages.
    """
    options_formatted = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
    
    system_content = (
        "You are a helpful AI assistant that answers multiple-choice reading comprehension questions. "
        "First, provide your reasoning step-by-step. "
        "Then, state the final answer in the format 'Answer: (X)', where X is the option letter."
    )
    
    user_content = (
        f"Read the following article and answer the question.\n\n"
        f"Article:\n{article}\n\n"
        f"Question: {question}\n"
        f"{options_formatted}\n\n"
        "Provide your reasoning and the final answer."
    )
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

def run_api_inference(
    model_name: str, 
    backend: str, 
    api_key: str, 
    base_url: Optional[str], 
    data_path: str, 
    output_path: str
):
    """
    Run inference using API-based models (OpenAI, Anthropic, DeepSeek).
    """
    
    # Client Initialization
    client = None
    if backend == "openai" or backend == "deepseek":
        if not openai:
            raise ImportError("OpenAI SDK is required. Install with `pip install openai`.")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
    elif backend == "anthropic":
        if not anthropic:
            raise ImportError("Anthropic SDK is required. Install with `pip install anthropic`.")
        client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        
    print(f"Loading data from {data_path}...")
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Normalize gold label
            if "gold_label" in data:
                try:
                    idx = int(data["gold_label"]) - 1
                    if 0 <= idx < 26:
                        data["gold_label"] = chr(65 + idx)
                except (ValueError, TypeError):
                    pass
            records.append(data)
            
    # Generation Loop
    print(f"Generating responses for {len(records)} samples using {backend} ({model_name})...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for record in tqdm(records):
            messages = format_quality_chat_messages(
                record["article"], 
                record["question"], 
                record["options"]
            )
            
            max_retries = 3
            output_text = ""
            
            for attempt in range(max_retries):
                try:
                    if backend in ["openai", "deepseek"]:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=0.0,
                            max_tokens=256
                        )
                        output_text = response.choices[0].message.content
                    elif backend == "anthropic":
                        system_msg = messages[0]["content"] if messages[0]["role"] == "system" else ""
                        chat_msgs = [m for m in messages if m["role"] != "system"]
                        
                        response = client.messages.create(
                            model=model_name,
                            system=system_msg,
                            messages=chat_msgs,
                            max_tokens=256,
                            temperature=0.0
                        )
                        output_text = response.content[0].text
                        
                    break # Success
                except Exception as e:
                    print(f"Error on attempt {attempt+1}: {e}")
                    time.sleep(2 ** attempt)
            
            if not output_text:
                print(f"Failed to generate for ID {record.get('question_unique_id')}")
                output_text = "ERROR"

            output_label, reason = extract_answer_components(output_text)
            
            final_record = {
                "question_unique_id": record.get("question_unique_id"),
                "article": record.get("article"),
                "gold_label": record.get("gold_label"),
                "output_label": output_label,
                "reason": reason,
                "raw_generation": output_text
            }
            f_out.write(json.dumps(final_record) + "\n")
            f_out.flush()
            
    print(f"Inference completed. Results saved to {output_path}")
