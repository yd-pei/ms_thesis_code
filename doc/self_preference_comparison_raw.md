# Self-Preference Comparison (Raw)

## Scope

- Before synonym replacement: `data/06_position_independent_response/raw`
- After synonym replacement: `data/13_position_independent_restyled/raw`

Pipeline executed:
1. Validity check on `data/12_output_restyled/raw` with `check_judge_output_values.py`
2. Position-independent filtering from `data/12_output_restyled/raw` to `data/13_position_independent_restyled/raw` with `filter_position_independent.py`
3. Self-preference analysis on both 13 (after) and 06 (before) with `analyze_self_preference.py`

---

## 1) Validity check for 12_output_restyled/raw

Source log: `outputs/check_12_output_restyled_raw.txt`

- files_checked: 8
- files_all_valid: 8
- total_records: 2372
- total_invalid: 0
- total_missing: 0

Conclusion: all `judge_output` values are valid (`'1'` or `'2'`).

---

## 2) Position-independent extraction to 13/raw

Source log: `outputs/filter_12_to_13_position_independent_raw.txt`

Per model:

| Model file | normal_total | kept | removed(position_dependent) |
|---|---:|---:|---:|
| llama31_70b | 355 | 350 | 5 |
| llama31_8b | 217 | 203 | 14 |
| llama33_70b | 292 | 279 | 13 |
| qwen25_7b | 322 | 306 | 16 |

Summary:
- pairs_processed: 4
- total_normal_records: 1186
- total_kept_position_independent: 1138
- total_position_dependent_removed: 48

---

## 3) Self-preference overall comparison (06 vs 13)

- 06 = before synonym replacement
- 13 = after synonym replacement

| Metric | 06 (before) | 13 (after) | Delta (after - before) |
|---|---:|---:|---:|
| files | 4 | 4 | 0 |
| total_records used | 1186 | 1138 | -48 |
| model_accuracy | 15.94% (189/1186) | 15.73% (179/1138) | -0.21 pp |
| self_selected_rate | 11.47% (136/1186) | 11.34% (129/1138) | -0.13 pp |
| harmful_self_preference_rate | 72.79% (99/136) | 72.87% (94/129) | +0.08 pp |

Overall reading:
- After replacement, dataset size reduced by 48 due to position-dependent removal in the new run.
- Overall self-selected rate decreases slightly.
- Harmful self-preference rate is nearly unchanged (very small increase).

---

## 4) Per-model comparison

### llama31_70b
- Before (06):
  - total_records: 355
  - model_accuracy: 18.03% (64/355)
  - self_selected_rate: 10.14% (36/355)
  - harmful_self_preference_rate: 69.44% (25/36)
- After (13):
  - total_records: 350
  - model_accuracy: 17.14% (60/350)
  - self_selected_rate: 10.29% (36/350)
  - harmful_self_preference_rate: 69.44% (25/36)

### llama31_8b
- Before (06):
  - total_records: 217
  - model_accuracy: 8.76% (19/217)
  - self_selected_rate: 3.69% (8/217)
  - harmful_self_preference_rate: 87.50% (7/8)
- After (13):
  - total_records: 203
  - model_accuracy: 8.87% (18/203)
  - self_selected_rate: 3.94% (8/203)
  - harmful_self_preference_rate: 87.50% (7/8)

### llama33_70b
- Before (06):
  - total_records: 292
  - model_accuracy: 21.92% (64/292)
  - self_selected_rate: 15.41% (45/292)
  - harmful_self_preference_rate: 60.00% (27/45)
- After (13):
  - total_records: 279
  - model_accuracy: 21.51% (60/279)
  - self_selected_rate: 14.70% (41/279)
  - harmful_self_preference_rate: 60.98% (25/41)

### qwen25_7b
- Before (06):
  - total_records: 322
  - model_accuracy: 13.04% (42/322)
  - self_selected_rate: 14.60% (47/322)
  - harmful_self_preference_rate: 85.11% (40/47)
- After (13):
  - total_records: 306
  - model_accuracy: 13.40% (41/306)
  - self_selected_rate: 14.38% (44/306)
  - harmful_self_preference_rate: 84.09% (37/44)

---

## 5) Raw logs

- `outputs/check_12_output_restyled_raw.txt`
- `outputs/filter_12_to_13_position_independent_raw.txt`
- `outputs/analyze_self_preference_06_raw.txt`
- `outputs/analyze_self_preference_13_raw.txt`
