import json
import os
from typing import Any

import torch
from tqdm import tqdm

from align_lab.inference.judge import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_messages,
    _find_reason_fields,
    _load_pairwise_records,
    _load_quality_context,
)


def run_raw_judge_inference(
    model_path: str,
    data_path: str,
    output_path: str,
    quality_path: str,
    swap_answers: bool = False,
    hf_token: str | None = None,
    batch_size: int = 1,
):
    """
    Judge pairwise model outputs using a single forward pass with transformers.

    Instead of generating text, this does one forward pass per sample and
    compares the logit/probability of token "1" vs "2" at the last position.
    Uses native bf16 precision (no quantization).

    Input:  cleaned pairwise JSONL from data/04_clean_pairwise_output.
    Output: same fields as input + judge_output, prob_1, prob_2.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # ── Load and validate pairwise data ──────────────────────────────────
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

    for idx, row in enumerate(records, 1):
        row_reason_fields = _find_reason_fields(row)
        if row_reason_fields != first_reason_fields:
            raise ValueError(
                f"Inconsistent reason fields at row {idx}. "
                f"Expected {first_reason_fields}, got {row_reason_fields}"
            )

    # ── Load article/question context ────────────────────────────────────
    target_qids = {row.get("question_unique_id") for row in records if row.get("question_unique_id")}
    print(f"Loading article/question context from {quality_path} for {len(target_qids)} questions...")
    quality_map = _load_quality_context(quality_path, target_qids)

    missing_qids = sorted(target_qids - set(quality_map.keys()))
    if missing_qids:
        print(f"Warning: {len(missing_qids)} question IDs not found in quality data. They will be skipped.")

    # ── Build judge messages ─────────────────────────────────────────────
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

    # ── Load model & tokenizer ───────────────────────────────────────────
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model in bf16: {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # ── Resolve token IDs for "1" and "2" ────────────────────────────────
    token_id_1 = tokenizer.encode("1", add_special_tokens=False)[0]
    token_id_2 = tokenizer.encode("2", add_special_tokens=False)[0]
    print(f"Token ID for '1': {token_id_1}, Token ID for '2': {token_id_2}")

    # ── Forward pass inference ───────────────────────────────────────────
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Running forward-pass judge for {len(all_messages)} samples (batch_size={batch_size})...")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for start in tqdm(range(0, len(all_messages), batch_size), desc="Judging"):
            end = min(start + batch_size, len(all_messages))
            batch_messages = all_messages[start:end]
            batch_rows = judge_records[start:end]

            # Apply chat template to each sample
            texts = [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in batch_messages
            ]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            # Move to the device of the model's first parameter
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # outputs.logits shape: (batch, seq_len, vocab_size)
            # For each sample, find the last non-padding position
            for i, row in enumerate(batch_rows):
                if "attention_mask" in inputs:
                    # Last non-padded position
                    seq_len = inputs["attention_mask"][i].sum().item() - 1
                else:
                    seq_len = inputs["input_ids"].shape[1] - 1

                last_logits = outputs.logits[i, seq_len, :]
                probs = torch.softmax(last_logits.float(), dim=-1)

                prob_1 = probs[token_id_1].item()
                prob_2 = probs[token_id_2].item()

                judge_output = "1" if prob_1 >= prob_2 else "2"

                out = dict(row)
                out["judge_output"] = judge_output
                out["prob_1"] = prob_1
                out["prob_2"] = prob_2
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Raw judge inference completed. Results saved to {output_path}")
