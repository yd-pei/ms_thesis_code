# Self-Preference Comparison (06/greedy vs 12/ds non-reverse)

## Scope

- Baseline: `data/06_position_independent_response/greedy` (4 files)
- Comparison: `data/12_output_restyled/ds` (top-level 4 files only; excludes `reverse/`)
- Metric script: `scripts/judge_analysis/analyze_self_preference.py`
- Harmful metric definition: `harmful_self_preference_rate = P(judge selects self | self model is wrong)`

Run commands:

```bash
/Users/yidingpei/Code/Research/ms_thesis_code/.venv/bin/python scripts/judge_analysis/analyze_self_preference.py --input-dir data/06_position_independent_response/greedy | tee outputs/analyze_self_preference_06_greedy.txt
/Users/yidingpei/Code/Research/ms_thesis_code/.venv/bin/python scripts/judge_analysis/analyze_self_preference.py --input-dir data/12_output_restyled/ds | tee outputs/analyze_self_preference_12_ds_non_reverse.txt
```

---

## Overall Comparison

| Metric | 06/greedy | 12/ds (non-reverse) | Delta (12 - 06) |
|---|---:|---:|---:|
| files | 4 | 4 | 0 |
| total_records used | 885 | 885 | 0 |
| objective_model_accuracy | 16.38% (145/885) | 16.38% (145/885) | +0.00 pp |
| judge_accuracy | 76.61% (678/885) | 74.12% (656/885) | -2.49 pp |
| self_selected_rate | 14.46% (128/885) | 17.63% (156/885) | +3.17 pp |
| harmful_self_preference_rate | 12.84% (95/740) | 16.22% (120/740) | +3.38 pp |

Reading:
- `judge_accuracy` decreases.
- `self_selected_rate` increases.
- `harmful_self_preference_rate` increases under the new conditional definition.

---

## Per-model Comparison

### llama31_70b
| Metric | 06/greedy | 12/ds | Delta (12 - 06) |
|---|---:|---:|---:|
| total_records | 236 | 236 | 0 |
| objective_model_accuracy | 21.19% (50/236) | 21.19% (50/236) | +0.00 pp |
| judge_accuracy | 75.00% (177/236) | 72.46% (171/236) | -2.54 pp |
| self_selected_rate | 12.29% (29/236) | 14.83% (35/236) | +2.54 pp |
| harmful_self_preference_rate | 10.22% (19/186) | 13.44% (25/186) | +3.22 pp |

### llama31_8b
| Metric | 06/greedy | 12/ds | Delta (12 - 06) |
|---|---:|---:|---:|
| total_records | 204 | 204 | 0 |
| objective_model_accuracy | 8.82% (18/204) | 8.82% (18/204) | +0.00 pp |
| judge_accuracy | 90.69% (185/204) | 84.80% (173/204) | -5.89 pp |
| self_selected_rate | 1.47% (3/204) | 7.35% (15/204) | +5.88 pp |
| harmful_self_preference_rate | 1.08% (2/186) | 7.53% (14/186) | +6.45 pp |

### llama33_70b
| Metric | 06/greedy | 12/ds | Delta (12 - 06) |
|---|---:|---:|---:|
| total_records | 242 | 242 | 0 |
| objective_model_accuracy | 21.90% (53/242) | 21.90% (53/242) | +0.00 pp |
| judge_accuracy | 65.29% (158/242) | 66.53% (161/242) | +1.24 pp |
| self_selected_rate | 27.69% (67/242) | 28.93% (70/242) | +1.24 pp |
| harmful_self_preference_rate | 25.93% (49/189) | 25.93% (49/189) | +0.00 pp |

### qwen25_7b
| Metric | 06/greedy | 12/ds | Delta (12 - 06) |
|---|---:|---:|---:|
| total_records | 203 | 203 | 0 |
| objective_model_accuracy | 11.82% (24/203) | 11.82% (24/203) | +0.00 pp |
| judge_accuracy | 77.83% (158/203) | 74.38% (151/203) | -3.45 pp |
| self_selected_rate | 14.29% (29/203) | 17.73% (36/203) | +3.44 pp |
| harmful_self_preference_rate | 13.97% (25/179) | 17.88% (32/179) | +3.91 pp |

---

## Logs

- `outputs/analyze_self_preference_06_greedy.txt`
- `outputs/analyze_self_preference_12_ds_non_reverse.txt`
