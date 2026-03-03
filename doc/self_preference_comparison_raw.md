# Self-Preference Comparison (Raw, 06 vs 12 non-reverse)

## Scope

- Baseline: `data/06_position_independent_response/raw` (4 files)
- Comparison: `data/12_output_restyled/raw` (top-level 4 files only; excludes `reverse/`)
- Metric script: `scripts/judge_analysis/analyze_self_preference.py` (updated to use `judge_accuracy`)
- Harmful metric definition: `harmful_self_preference_rate = P(judge selects self | self model is wrong)`

Run commands:

```bash
/Users/yidingpei/Code/Research/ms_thesis_code/.venv/bin/python scripts/judge_analysis/analyze_self_preference.py --input-dir data/06_position_independent_response/raw | tee outputs/analyze_self_preference_06_raw.txt
/Users/yidingpei/Code/Research/ms_thesis_code/.venv/bin/python scripts/judge_analysis/analyze_self_preference.py --input-dir data/12_output_restyled/raw | tee outputs/analyze_self_preference_12_raw_non_reverse.txt
```

---

## Overall Comparison

| Metric | 06/raw | 12/raw (non-reverse) | Delta (12 - 06) |
|---|---:|---:|---:|
| files | 4 | 4 | 0 |
| total_records used | 1186 | 1186 | 0 |
| objective_model_accuracy | 15.94% (189/1186) | 15.94% (189/1186) | +0.00 pp |
| judge_accuracy | 78.84% (935/1186) | 76.90% (912/1186) | -1.94 pp |
| self_selected_rate | 11.47% (136/1186) | 13.74% (163/1186) | +2.27 pp |
| harmful_self_preference_rate | 9.93% (99/997) | 12.44% (124/997) | +2.51 pp |

Reading:
- On this 06 vs 12 comparison, `judge_accuracy` decreases.
- `self_selected_rate` increases.
- `harmful_self_preference_rate` also increases.

---

## Per-model Comparison

### llama31_70b
| Metric | 06/raw | 12/raw | Delta (12 - 06) |
|---|---:|---:|---:|
| total_records | 355 | 355 | 0 |
| objective_model_accuracy | 18.03% (64/355) | 18.03% (64/355) | +0.00 pp |
| judge_accuracy | 78.03% (277/355) | 78.59% (279/355) | +0.56 pp |
| self_selected_rate | 10.14% (36/355) | 11.27% (40/355) | +1.13 pp |
| harmful_self_preference_rate | 8.59% (25/291) | 8.93% (26/291) | +0.34 pp |

### llama31_8b
| Metric | 06/raw | 12/raw | Delta (12 - 06) |
|---|---:|---:|---:|
| total_records | 217 | 217 | 0 |
| objective_model_accuracy | 8.76% (19/217) | 8.76% (19/217) | +0.00 pp |
| judge_accuracy | 88.48% (192/217) | 82.95% (180/217) | -5.53 pp |
| self_selected_rate | 3.69% (8/217) | 10.14% (22/217) | +6.45 pp |
| harmful_self_preference_rate | 3.54% (7/198) | 10.10% (20/198) | +6.56 pp |

### llama33_70b
| Metric | 06/raw | 12/raw | Delta (12 - 06) |
|---|---:|---:|---:|
| total_records | 292 | 292 | 0 |
| objective_model_accuracy | 21.92% (64/292) | 21.92% (64/292) | +0.00 pp |
| judge_accuracy | 75.00% (219/292) | 73.29% (214/292) | -1.71 pp |
| self_selected_rate | 15.41% (45/292) | 15.75% (46/292) | +0.34 pp |
| harmful_self_preference_rate | 11.84% (27/228) | 13.16% (30/228) | +1.32 pp |

### qwen25_7b
| Metric | 06/raw | 12/raw | Delta (12 - 06) |
|---|---:|---:|---:|
| total_records | 322 | 322 | 0 |
| objective_model_accuracy | 13.04% (42/322) | 13.04% (42/322) | +0.00 pp |
| judge_accuracy | 76.71% (247/322) | 74.22% (239/322) | -2.49 pp |
| self_selected_rate | 14.60% (47/322) | 17.08% (55/322) | +2.48 pp |
| harmful_self_preference_rate | 14.29% (40/280) | 17.14% (48/280) | +2.85 pp |

---

## Logs

- `outputs/analyze_self_preference_06_raw.txt`
- `outputs/analyze_self_preference_12_raw_non_reverse.txt`
