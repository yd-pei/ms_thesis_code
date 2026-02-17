
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


def run_inference(model_path):
    pass



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

def run_offline_inference(
    model_path: str, 
    data_path: str, 
    output_path: str, 
    hf_token: str = None, 
    quantization: str = "bitsandbytes"
):
    """
    Run offline inference using vLLM on QuALITY dataset.
    
    Args:
        model_path: HuggingFace model ID or local path (use an Instruct/chat model).
        data_path: Path to the input JSONL file.
        output_path: Path to save inference results.
        hf_token: HuggingFace token for gated models.
        quantization: Quantization method. Options: 'bitsandbytes' (8-bit, recommended
                      for memory-constrained setups), 'gptq', 'awq', 'fp8', None.
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
    all_messages = []
    records = []
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            messages = format_quality_chat_messages(
                data["article"],
                data["question"],
                data["options"]
            )
            all_messages.append(messages)
            
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
    print(f"Initializing vLLM with model: {model_path} (quantization={quantization})")
    
    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
    )
    
    if quantization == "bitsandbytes":
        # 8-bit quantization via bitsandbytes: loads weights in 8-bit directly,
        # significantly reducing VRAM usage (~70GB -> ~35GB for 70B model)
        llm_kwargs.update(
            dtype="bfloat16",
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            gpu_memory_utilization=0.92,
        )
    elif quantization in ("gptq", "awq"):
        # Pre-quantized models (model_path must point to a GPTQ/AWQ checkpoint)
        llm_kwargs.update(
            dtype="half",
            quantization=quantization,
        )
    elif quantization == "fp8":
        # FP8 quantization (requires H100/Ada GPU)
        llm_kwargs.update(
            dtype="bfloat16",
            quantization="fp8",
        )
    else:
        # No quantization, full precision
        llm_kwargs["dtype"] = "bfloat16"
    
    llm = LLM(**llm_kwargs)
    
    # 3. Define Sampling Params
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.90,
        max_tokens=2048,
        n=1,
    )

    # 4. Generate (chat format for Instruct models)
    print(f"Generating responses for {len(all_messages)} samples...")
    outputs = llm.chat(all_messages, sampling_params)

    # 5. Save Results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving results to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record, output in zip(records, outputs):
            generated_text = output.outputs[0].text.strip()
            
            output_label, reason = extract_answer_components(generated_text)
            
            # Construct the final output record
            final_record = {
                "article": record.get("article"),
                "question_unique_id": record.get("question_unique_id"),
                "gold_label": record.get("gold_label"),
                "output_label": output_label,
                "reason": reason,
                "raw_generation": generated_text # keeping raw output for debugging
            }
            f.write(json.dumps(final_record) + "\n")
    
    print("Inference completed.")


def format_quality_chat_messages(article: str, question: str, options: List[str]) -> List[dict]:
    """
    Format the input for chat-based LLMs (OpenAI/Claude/DeepSeek/vLLM-Instruct) using messages.
    Zero-shot format: system sets the task; user provides article, question, and options.
    """
    options_formatted = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
    
    system_content = (
        "You are a helpful AI assistant that answers multiple-choice reading comprehension questions.\n"
        "First, provide your reasoning step-by-step.\n"
        "Then, state the final answer on a new line in the exact format:\n"
        "Answer: (X)\n"
        "where X is one of A, B, C, or D."
    )
    
    user_content = (
        f"Read the following article and answer the question.\n\n"
        f"Article:\n{article}\n\n"
        f"Question: {question}\n"
        f"{options_formatted}"
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
                            temperature=1.0,
                            top_p=0.90,
                            max_tokens=2048,
                            n=1,
                        )
                        output_text = response.choices[0].message.content
                    elif backend == "anthropic":
                        system_msg = messages[0]["content"] if messages[0]["role"] == "system" else ""
                        chat_msgs = [m for m in messages if m["role"] != "system"]
                        
                        response = client.messages.create(
                            model=model_name,
                            system=system_msg,
                            messages=chat_msgs,
                            max_tokens=2048,
                            temperature=1.0,
                            top_p=0.90,
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
