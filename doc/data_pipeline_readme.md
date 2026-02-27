# Data Pipeline README

This document maps each `data/` stage to the script(s) used to generate it.

## Stage Mapping

### `data/01_processed_quality/`
- Purpose: Processed QuALITY dataset in JSONL format.
- Main generator:
  - `scripts/test/preprocess_quality.py` (dataset preprocessing script used in this repo history).

### `data/02_quality_response/`
- Purpose: Model answer generation outputs on QuALITY.
- Main generator:
  - CLI inference pipeline (`align_lab inference`) in `src/align_lab/main.py`.
- Related script:
  - `scripts/eval_inference_accuracy.py` (evaluation, not generation).

### `data/03_pairwise_output/`
- Purpose: Merge model outputs into pairwise comparison records.
- Main generator:
  - `scripts/merge_pairwise_responses.py`

### `data/04_clean_pairwise_output/`
- Purpose: Cleaned pairwise records.
- Main generator:
  - `scripts/clean_pairwise_outputs.py`

### `data/05_pairwise_response/`
- Purpose: Judge model outputs on pairwise samples (normal and reverse).
- Main generator:
  - CLI judge pipeline (`align_lab judge`) in `src/align_lab/main.py`.

### `data/06_position_independent_response/`
- Purpose: Keep only position-independent judged samples.
- Main generator:
  - `scripts/judge_analysis/filter_position_independent.py`

### `data/07_extracted_prompt/`
- Purpose: Extract fields for restyle prompt construction.
- Main generator:
  - `scripts/restyle/extract_prompt_fields.py`

### `data/08_gemini_api/`
- Purpose: Gemini Batch API request JSONL files.
- Main generator:
  - `scripts/judge_analysis/prepare_gemini_synonym_batch_input.py`

### `data/09_gemini_restyle_output/`
- Purpose: Gemini batch outputs and batch tracking metadata.
- Main scripts:
  - Upload input files: `scripts/restyle/upload2fileAPI.py`
  - Create batch jobs: `scripts/restyle/create_gemini_batch_jobs.py`
  - Poll and download outputs: `scripts/restyle/poll_and_download_gemini_batches.py`
- Tracking files:
  - `with_ds/batch_jobs.json`
  - `with_ds/batch_receive_state.json`

### `data/10_replaced_answer/`
- Purpose: Synonym-replaced answer-only outputs.
- Main generator:
  - `scripts/restyle/replace_answers_with_synonyms.py`

### `data/11_restyled_position_independent_response/`
- Purpose: Restyled version of position-independent judge files.
- Main generator:
  - `scripts/restyle/replace_reason_in_position_independent.py --output-dir data/11_restyled_position_independent_response/greedy`
- Notes:
  - One-to-one replacement by `question_unique_id`
  - Only `<model>_output_reason` is replaced from `data/10_replaced_answer/with_ds`
  - All other fields are preserved.

## Post-Restyle Injection (Recommended: write to 11)

If you want to inject restyled answers back into position-independent judge files:
- Script:
  - `scripts/restyle/replace_reason_in_position_independent.py`
- Behavior:
  - One-to-one match by `question_unique_id`
  - Replace only `<model>_output_reason`
  - Keep all other fields unchanged.
- Recommended output target:
  - `data/11_restyled_position_independent_response/greedy`

## Typical Command Sequence (Restyle branch)

1. Extract prompt fields:
   - `uv run python scripts/restyle/extract_prompt_fields.py`
2. Build Gemini batch input:
   - `uv run python scripts/judge_analysis/prepare_gemini_synonym_batch_input.py`
3. Upload files:
   - `uv run python scripts/restyle/upload2fileAPI.py`
4. Create batch jobs:
   - `uv run python scripts/restyle/create_gemini_batch_jobs.py --model gemini-2.5-flash`
5. Poll and download:
   - `uv run python scripts/restyle/poll_and_download_gemini_batches.py`
6. Build replaced-answer files:
   - `uv run python scripts/restyle/replace_answers_with_synonyms.py`
7. Inject replaced answers into `11`:
  - `uv run python scripts/restyle/replace_reason_in_position_independent.py --output-dir data/11_restyled_position_independent_response/greedy`
