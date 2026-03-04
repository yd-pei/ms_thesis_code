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

## Key Observations (Before Restyle)

1. **Self 远强于 opponent 时** (llama33_70b vs llama31_8b): self-select rate 极高 (92.6%)，但 judge accuracy 也高 (81.2%)，大部分选择自己是合理的；但 harmful rate 仍达 88.1%，说明即使自己答错也强烈偏向自己。

2. **Opponent 远强于 self 时** (llama31_8b vs ds_v3): self-select rate 极低 (1.0%)，judge 几乎完全选择 opponent，说明质量差距大时 judge 能正确识别。

3. **实力接近时** (llama31_8b vs gpt_oss, judge acc 51.5%): 接近随机选择，self-select rate 21.1%，说明 judge 在无法区分质量时没有明显偏向自己。

4. **Harmful rate 最高的组合**: llama33_70b vs llama31_8b (88.1%) 和 llama33_70b vs mistral (73.3%)，说明 llama33_70b 作为 judge 时有较强的自偏好倾向。

5. **qwen25_7b 自偏好模式**: 对 ds_v3 (强对手) harmful rate 低 (11.7%)，对 mistral (弱对手) harmful rate 高 (62.5%)，说明自偏好程度与实力对比相关。

---

## Restyle 前后对比

Pipeline: `065/before_restyle` 中主模型的 `output_reason` 经 Gemini 同义词替换（每条回答替换 2 个非停用词为同义词）后重新由同一 judge 评判，结果写入 `065/after_restyle`。

### Overall 对比

| Metric | Before | After | Δ |
|--------|-------:|------:|--:|
| judge_accuracy | 68.68% (1704/2481) | 67.59% (1677/2481) | **−1.09%** |
| self_selected_rate | 43.73% (1085/2481) | 45.30% (1124/2481) | **+1.57%** |
| harmful_self_preference_rate | 26.21% (341/1301) | 28.75% (374/1301) | **+2.54%** |

### Per-pair 对比

| Judge vs Opponent | Samples | Judge Acc (Before) | Judge Acc (After) | Δ | Harmful (Before) | Harmful (After) | Δ |
|-------------------|--------:|-------------------:|------------------:|--:|-----------------:|-----------------:|--:|
| llama33_70b vs llama31_8b | 394 | 81.22% | 81.22% | 0.00 | 88.14% (52/59) | 88.14% (52/59) | 0.00 |
| llama31_8b vs ds_v3 | 405 | 83.95% | 80.99% | −2.96 | 1.16% (4/344) | 5.23% (18/344) | **+4.07** |
| qwen25_7b vs ds_v3 | 263 | 76.43% | 72.24% | −4.19 | 11.66% (26/223) | 17.94% (40/223) | **+6.28** |
| llama31_70b vs qwen25_7b | 370 | 69.73% | 70.00% | +0.27 | 59.30% (51/86) | 58.14% (50/86) | −1.16 |
| llama33_70b vs ds_v3 | 162 | 64.20% | 63.58% | −0.62 | 12.24% (12/98) | 13.27% (13/98) | +1.03 |
| llama33_70b vs mistral | 210 | 62.38% | 60.95% | −1.43 | 73.33% (44/60) | 75.00% (45/60) | +1.67 |
| llama31_70b vs ds_v3 | 234 | 61.54% | 60.68% | −0.86 | 15.72% (25/159) | 16.98% (27/159) | +1.26 |
| llama31_8b vs gpt_oss | 204 | 51.47% | 50.98% | −0.49 | 17.71% (17/96) | 21.88% (21/96) | +4.17 |
| qwen25_7b vs mistral | 239 | 42.26% | 43.10% | +0.84 | 62.50% (110/176) | 61.36% (108/176) | −1.14 |

### Self-Select Rate 对比

| Judge vs Opponent | Self-Select (Before) | Self-Select (After) | Δ |
|-------------------|---------------------:|--------------------:|--:|
| llama33_70b vs llama31_8b | 92.64% | 92.64% | 0.00 |
| llama31_8b vs ds_v3 | 0.99% | 4.94% | **+3.95** |
| qwen25_7b vs ds_v3 | 11.41% | 17.87% | **+6.46** |
| llama31_70b vs qwen25_7b | 74.05% | 73.78% | −0.27 |
| llama33_70b vs ds_v3 | 18.52% | 19.14% | +0.62 |
| llama33_70b vs mistral | 75.71% | 75.24% | −0.47 |
| llama31_70b vs ds_v3 | 14.96% | 15.81% | +0.85 |
| llama31_8b vs gpt_oss | 21.08% | 24.51% | +3.43 |
| qwen25_7b vs mistral | 60.67% | 59.83% | −0.84 |

### Key Observations (Restyle 对比)

1. **Restyle 未降低自偏好**：整体 harmful rate 从 26.21% 上升到 28.75%（+2.54%），self-select rate 从 43.73% 上升到 45.30%（+1.57%），同义词替换没有起到缓解自偏好的作用。

2. **Judge accuracy 普遍微降**：整体从 68.68% 降至 67.59%（−1.09%），7/9 组合出现下降，说明同义词替换可能轻微破坏了回答的可读性或信息表达，反而干扰了 judge 的正确判断。

3. **弱模型 vs 强对手受影响最大**：llama31_8b vs ds_v3（harmful +4.07%）和 qwen25_7b vs ds_v3（harmful +6.28%）变化最大。这些组合中 self 的客观正确率很低（~15%），restyle 后 judge 反而更多地选择了质量更差的 self 回答。

4. **强模型 vs 弱对手几乎无变化**：llama33_70b vs llama31_8b 的三项指标完全不变（judge acc 81.22%, self-select 92.64%, harmful 88.14%），说明当 judge 对自身输出的偏好已经极端饱和时，微小的文本风格变化无法产生影响。

5. **少数组合有微弱改善**：llama31_70b vs qwen25_7b（harmful −1.16%）和 qwen25_7b vs mistral（harmful −1.14%）出现了微弱的 harmful rate 下降，但变化幅度很小，不具有统计显著性。

6. **总体结论**：仅替换 2 个非停用词为同义词的 restyle 强度不足以改变 judge 的自偏好行为。自偏好可能更多来自内容层面（推理结构、论证方式）而非词汇层面的风格特征。
