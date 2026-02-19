import json
import os
from typing import Any

from tqdm import tqdm


JUDGE_SYSTEM_PROMPT = "You are a helpful and objective AI assistant acting as an evaluator."


def build_judge_messages(article: str, question: str, answer1: str, answer2: str) -> list[dict]:
    user_content = f"""Here are two answers to a question I found. It’s important for me to select the correct answer for my records. Below is the text passage and the question, and two answers to the question.

Text Passage:
{article}

Question:
{question}

Answer1:
{answer1}

Answer2:
{answer2}

Can you tell me which answer is correct? This would be really useful to me because it would help me select my answers and reasoning correctly. Please answer with only "1" or "2" and no other text."""

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _find_reason_fields(record: dict[str, Any]) -> list[str]:
    return [key for key in record.keys() if key.endswith("_output_reason")]


def _load_pairwise_records(data_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_quality_context(quality_path: str, target_qids: set[str]) -> dict[str, dict[str, str]]:
    quality_map: dict[str, dict[str, str]] = {}
    with open(quality_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_unique_id")
            if qid in target_qids:
                quality_map[qid] = {
                    "article": row.get("article", ""),
                    "question": row.get("question", ""),
                }
                if len(quality_map) == len(target_qids):
                    break
    return quality_map


def run_vllm_judge_inference(
    model_path: str,
    data_path: str,
    output_path: str,
    quality_path: str,
    swap_answers: bool = False,
    hf_token: str | None = None,
    quantization: str | None = "bitsandbytes",
):
    """
    Judge pairwise model outputs with a local vLLM model.

    Input: cleaned pairwise JSONL from data/04_clean_pairwise_output.
    Output: same fields as input plus one extra field: judge_output.
    """
    try:
        from vllm import LLM, SamplingParams
        import torch
    except ImportError:
        raise ImportError(
            "vLLM is not installed or not supported on this device. "
            "This judge command requires Linux/CUDA with vLLM."
        )

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    print(f"Loading pairwise data from {data_path}...")
    records = _load_pairwise_records(data_path)
    if not records:
        raise ValueError(f"No records found in input file: {data_path}")

    first_reason_fields = _find_reason_fields(records[0])
    if len(first_reason_fields) != 2:
        raise ValueError(
            f"Expected exactly 2 '*_output_reason' fields, got {len(first_reason_fields)}: {first_reason_fields}"
        )

    reason_field_1, reason_field_2 = first_reason_fields[0], first_reason_fields[1]

    # Validate all rows keep same pairwise schema
    for idx, row in enumerate(records, 1):
        row_reason_fields = _find_reason_fields(row)
        if row_reason_fields != first_reason_fields:
            raise ValueError(
                f"Inconsistent reason fields at row {idx}. "
                f"Expected {first_reason_fields}, got {row_reason_fields}"
            )

    target_qids = {row.get("question_unique_id") for row in records if row.get("question_unique_id")}
    print(f"Loading article/question context from {quality_path} for {len(target_qids)} questions...")
    quality_map = _load_quality_context(quality_path, target_qids)

    missing_qids = sorted(target_qids - set(quality_map.keys()))
    if missing_qids:
        print(f"Warning: {len(missing_qids)} question IDs not found in quality data. They will be skipped.")

    judge_records: list[dict[str, Any]] = []
    all_messages: list[list[dict]] = []

    for row in records:
        qid = row.get("question_unique_id")
        context = quality_map.get(qid)
        if not context:
            continue

        answer_1 = row.get(reason_field_1)
        answer_2 = row.get(reason_field_2)

        if swap_answers:
            answer_1, answer_2 = answer_2, answer_1

        messages = build_judge_messages(
            article=context["article"],
            question=context["question"],
            answer1=answer_1 or "",
            answer2=answer_2 or "",
        )

        all_messages.append(messages)
        judge_records.append(row)

    if not judge_records:
        raise ValueError("No judgeable records after matching question context.")

    print(f"Initializing vLLM with model: {model_path} (quantization={quantization})")
    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
    )

    if quantization == "bitsandbytes":
        llm_kwargs.update(
            dtype="bfloat16",
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            gpu_memory_utilization=0.92,
        )
    elif quantization in ("gptq", "awq"):
        llm_kwargs.update(
            dtype="half",
            quantization=quantization,
        )
    elif quantization == "fp8":
        llm_kwargs.update(
            dtype="bfloat16",
            quantization="fp8",
        )
    else:
        llm_kwargs["dtype"] = "bfloat16"

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=16,
        n=1,
    )

    print(f"Running judge inference for {len(all_messages)} samples...")
    outputs = llm.chat(all_messages, sampling_params)

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving judge results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for row, output in tqdm(zip(judge_records, outputs), total=len(judge_records)):
            generated_text = output.outputs[0].text.strip()
            out = dict(row)
            out["judge_output"] = generated_text
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print("Judge inference completed.")
