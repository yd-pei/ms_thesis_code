# Self Preference

## Metric Definition

- model_accuracy: self model output_label matches gold_label
- self_selected_rate: judge selects answer1/self
- harmful_self_preference_rate: among self-selected samples, proportion where self model is wrong

## Restyle Before (Baseline)

Source: original position-independent evaluation outputs.

| Model | Total | model_accuracy | self_selected_rate | harmful_self_preference_rate |
|---|---:|---:|---:|---:|
| llama31_70b | 244 | 20.49% (50/244) | 18.85% (46/244) | 67.39% (31/46) |
| llama31_8b | 192 | 7.29% (14/192) | 7.81% (15/192) | 93.33% (14/15) |
| llama33_70b | 233 | 19.74% (46/233) | 30.90% (72/233) | 70.83% (51/72) |
| qwen25_7b | 205 | 12.68% (26/205) | 14.15% (29/205) | 89.66% (26/29) |

### Overall (Before)

- total_records: 874
- model_accuracy: 15.56% (136/874)
- self_selected_rate: 18.54% (162/874)
- harmful_self_preference_rate: 75.31% (122/162)

## Restyle After

Source: `data/13_position_independent_restyled/ds` with command:

`uv run python scripts/judge_analysis/analyze_self_preference.py --input-dir data/13_position_independent_restyled/ds`

| Model | Total | model_accuracy | self_selected_rate | harmful_self_preference_rate |
|---|---:|---:|---:|---:|
| llama31_70b | 226 | 21.68% (49/226) | 11.95% (27/226) | 66.67% (18/27) |
| llama31_8b | 189 | 9.52% (18/189) | 0.53% (1/189) | 0.00% (0/1) |
| llama33_70b | 227 | 21.59% (49/227) | 26.87% (61/227) | 70.49% (43/61) |
| qwen25_7b | 189 | 12.70% (24/189) | 12.70% (24/189) | 83.33% (20/24) |

### Overall (After)

- total_records: 831
- model_accuracy: 16.85% (140/831)
- self_selected_rate: 13.60% (113/831)
- harmful_self_preference_rate: 71.68% (81/113)

## Delta (After - Before)

- total_records: -43
- model_accuracy: +1.29 pp
- self_selected_rate: -4.94 pp
- harmful_self_preference_rate: -3.63 pp

## Notes

- Before/After sample sizes differ (874 vs 831), so the comparison is directional rather than fully matched-pair causal.
- On this aggregate view, restyle reduces self-selection and harmful self-preference rate.

## Restyle Direct (12/ds, non-reverse)

Source: `data/12_output_restyled/ds` (4 files; excludes reverse folder) with command:

`for f in data/12_output_restyled/ds/*.jsonl; do uv run python scripts/judge_analysis/analyze_self_preference.py --input-file "$f"; done`

| Model | Total | model_accuracy | self_selected_rate | harmful_self_preference_rate |
|---|---:|---:|---:|---:|
| llama31_70b | 236 | 21.19% (50/236) | 14.83% (35/236) | 71.43% (25/35) |
| llama31_8b | 204 | 8.82% (18/204) | 7.35% (15/204) | 93.33% (14/15) |
| llama33_70b | 242 | 21.90% (53/242) | 28.93% (70/242) | 70.00% (49/70) |
| qwen25_7b | 203 | 11.82% (24/203) | 17.73% (36/203) | 88.89% (32/36) |

### Overall (12/ds Direct)

- total_records: 885
- model_accuracy: 16.38% (145/885)
- self_selected_rate: 17.63% (156/885)
- harmful_self_preference_rate: 76.92% (120/156)

### Delta vs Restyle Before (06/greedy baseline)

- total_records: +11
- model_accuracy: +0.82 pp
- self_selected_rate: -0.91 pp
- harmful_self_preference_rate: +1.61 pp

### Log

- `outputs/analyze_self_preference_12_ds_non_reverse.txt`