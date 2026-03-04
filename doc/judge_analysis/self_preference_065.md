# Self-Preference Analysis (065 Position-Independent)

## Scope

- Pipeline: `02/raw → 025 → 035 → 045 → 055 → 065`
- Data: `data/065_position_independent/` (9 files, multi-pair)
- Metric script: `scripts/judge_analysis/analyze_self_preference.py`
- Metric definitions:
  - **judge_accuracy**: judge 选中的回答与 gold_label 一致的比例
  - **self_selected_rate**: judge 选择自己回答 (answer 1) 的比例
  - **harmful_self_preference_rate**: self 答错的样本中，judge 仍选自己的比例

Run command:

```bash
uv run python scripts/judge_analysis/analyze_self_preference.py --input-dir data/065_position_independent
```

---

## Overall

| Metric | Value |
|--------|------:|
| files | 9 |
| total_records | 2481 |
| judge_accuracy | 68.68% (1704/2481) |
| self_selected_rate | 43.73% (1085/2481) |
| harmful_self_preference_rate | 26.21% (341/1301) |

---

## Per-pair Results

### Position-Independent Filtering (055 → 065)

| Pair | Before | After | Position-Dependent Removed |
|------|-------:|------:|---------------------------:|
| llama31_70b vs ds_v3 | 320 | 234 | 86 (26.9%) |
| llama31_70b vs qwen25_7b | 541 | 370 | 171 (31.6%) |
| llama31_8b vs ds_v3 | 590 | 405 | 185 (31.4%) |
| llama31_8b vs gpt_oss | 472 | 204 | 268 (56.8%) |
| llama33_70b vs ds_v3 | 289 | 162 | 127 (43.9%) |
| llama33_70b vs llama31_8b | 532 | 394 | 138 (25.9%) |
| llama33_70b vs mistral | 357 | 210 | 147 (41.2%) |
| qwen25_7b vs ds_v3 | 567 | 263 | 304 (53.6%) |
| qwen25_7b vs mistral | 533 | 239 | 294 (55.2%) |

### Self-Preference Metrics

| Judge vs Opponent | Samples | Self Obj Acc | Judge Acc | Self-Select Rate | Harmful Rate |
|-------------------|--------:|-------------:|----------:|-----------------:|-------------:|
| llama33_70b vs llama31_8b | 394 | 85.03% (335/394) | 81.22% (320/394) | 92.64% (365/394) | 88.14% (52/59) |
| llama31_8b vs ds_v3 | 405 | 15.06% (61/405) | 83.95% (340/405) | 0.99% (4/405) | 1.16% (4/344) |
| qwen25_7b vs ds_v3 | 263 | 15.21% (40/263) | 76.43% (201/263) | 11.41% (30/263) | 11.66% (26/223) |
| llama31_70b vs qwen25_7b | 370 | 76.76% (284/370) | 69.73% (258/370) | 74.05% (274/370) | 59.30% (51/86) |
| llama33_70b vs ds_v3 | 162 | 39.51% (64/162) | 64.20% (104/162) | 18.52% (30/162) | 12.24% (12/98) |
| llama33_70b vs mistral | 210 | 71.43% (150/210) | 62.38% (131/210) | 75.71% (159/210) | 73.33% (44/60) |
| llama31_70b vs ds_v3 | 234 | 32.05% (75/234) | 61.54% (144/234) | 14.96% (35/234) | 15.72% (25/159) |
| llama31_8b vs gpt_oss | 204 | 52.94% (108/204) | 51.47% (105/204) | 21.08% (43/204) | 17.71% (17/96) |
| qwen25_7b vs mistral | 239 | 26.36% (63/239) | 42.26% (101/239) | 60.67% (145/239) | 62.50% (110/176) |

---

## Key Observations

1. **Self 远强于 opponent 时** (llama33_70b vs llama31_8b): self-select rate 极高 (92.6%)，但 judge accuracy 也高 (81.2%)，大部分选择自己是合理的；但 harmful rate 仍达 88.1%，说明即使自己答错也强烈偏向自己。

2. **Opponent 远强于 self 时** (llama31_8b vs ds_v3): self-select rate 极低 (1.0%)，judge 几乎完全选择 opponent，说明质量差距大时 judge 能正确识别。

3. **实力接近时** (llama31_8b vs gpt_oss, judge acc 51.5%): 接近随机选择，self-select rate 21.1%，说明 judge 在无法区分质量时没有明显偏向自己。

4. **Harmful rate 最高的组合**: llama33_70b vs llama31_8b (88.1%) 和 llama33_70b vs mistral (73.3%)，说明 llama33_70b 作为 judge 时有较强的自偏好倾向。

5. **qwen25_7b 自偏好模式**: 对 ds_v3 (强对手) harmful rate 低 (11.7%)，对 mistral (弱对手) harmful rate 高 (62.5%)，说明自偏好程度与实力对比相关。
