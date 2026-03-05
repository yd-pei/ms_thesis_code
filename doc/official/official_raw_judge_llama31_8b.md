# Official QuALITY Raw Judge Analysis (llama31_8b only)

This report uses the same metric definitions as `scripts/judge_analysis/analyze_self_preference.py`.

- `judge_accuracy`: judge-selected answer label matches `gold_label`
- `harmful_self_preference_rate`: among samples where self model is wrong, judge still selects self (`judge_output==1`)
- `self_selected_rate`: rate of selecting self answer

Data sources:
- Raw outputs: `/Users/yidingpei/Code/Research/ms_thesis_code/data/official/judged_before`, `/Users/yidingpei/Code/Research/ms_thesis_code/data/official/judged_after`
- Position-independent filtered outputs: `/Users/yidingpei/.codex/worktrees/2627/ms_thesis_code/outputs/official_analysis/position_independent/before`, `/Users/yidingpei/.codex/worktrees/2627/ms_thesis_code/outputs/official_analysis/position_independent/after`

## Unfiltered Raw Judge (pairwise files, excluding `official_mix`)

| Phase | Files | Records | Judge Accuracy | Self-Selected Rate | Harmful Self-Preference |
|---|---:|---:|---:|---:|---:|
| before | 5 | 3490 | 57.34% (2001/3490) | 63.07% (2201/3490) | 54.01% (1347/2494) |
| after | 5 | 3490 | 58.88% (2055/3490) | 60.43% (2109/3490) | 51.08% (1274/2494) |

Delta (after - before): judge_accuracy +1.55 pp, self_selected_rate -2.64 pp, harmful_rate -2.93 pp, records +0.

### Unfiltered Raw Judge Per-file (pairwise only)

| File | Phase | Records | Judge Accuracy | Harmful Self-Preference |
|---|---|---:|---:|---:|
| `llama31_8b__ds_v3__raw_judge.jsonl` | before | 782 | 53.71% (420/782) | 53.12% (341/642) |
| `llama31_8b__llama4_maverick_17b__raw_judge.jsonl` | before | 699 | 56.94% (398/699) | 52.32% (271/518) |
| `llama31_8b__llama4_scout_17b__raw_judge.jsonl` | before | 655 | 61.83% (405/655) | 48.41% (213/440) |
| `llama31_8b__qwen25_72b__raw_judge.jsonl` | before | 699 | 55.51% (388/699) | 53.32% (281/527) |
| `llama31_8b__qwen25_7b__raw_judge.jsonl` | before | 655 | 59.54% (390/655) | 65.67% (241/367) |
| `llama31_8b__ds_v3__raw_judge.jsonl` | after | 782 | 54.35% (425/782) | 51.56% (331/642) |
| `llama31_8b__llama4_maverick_17b__raw_judge.jsonl` | after | 699 | 58.94% (412/699) | 49.42% (256/518) |
| `llama31_8b__llama4_scout_17b__raw_judge.jsonl` | after | 655 | 63.66% (417/655) | 43.86% (193/440) |
| `llama31_8b__qwen25_72b__raw_judge.jsonl` | after | 699 | 58.66% (410/699) | 49.34% (260/527) |
| `llama31_8b__qwen25_7b__raw_judge.jsonl` | after | 655 | 59.69% (391/655) | 63.76% (234/367) |

### Unfiltered Raw Judge Mix-only (`official_mix`)

| Phase | Files | Records | Judge Accuracy | Harmful Self-Preference |
|---|---:|---:|---:|---:|
| before | 1 | 3490 | 57.42% (2004/3490) | 53.97% (1346/2494) |
| after | 1 | 3490 | 58.94% (2057/3490) | 51.00% (1272/2494) |

## Position-Independent Raw Judge (pairwise files, excluding `official_mix`)

| Phase | Files | Records | Judge Accuracy | Self-Selected Rate | Harmful Self-Preference |
|---|---:|---:|---:|---:|---:|
| before | 5 | 1474 | 85.62% (1262/1474) | 13.03% (192/1474) | 5.86% (71/1212) |
| after | 5 | 1557 | 85.23% (1327/1557) | 11.88% (185/1557) | 5.53% (71/1284) |

Delta (after - before): judge_accuracy -0.39 pp, self_selected_rate -1.14 pp, harmful_rate -0.33 pp, records +83.

### Position-Independent Raw Judge Per-file (pairwise only)

| File | Phase | Records | Judge Accuracy | Harmful Self-Preference |
|---|---|---:|---:|---:|
| `llama31_8b__ds_v3__raw_judge.jsonl` | before | 351 | 89.46% (314/351) | 5.36% (17/317) |
| `llama31_8b__llama4_maverick_17b__raw_judge.jsonl` | before | 307 | 86.32% (265/307) | 4.65% (12/258) |
| `llama31_8b__llama4_scout_17b__raw_judge.jsonl` | before | 299 | 83.61% (250/299) | 5.02% (12/239) |
| `llama31_8b__qwen25_72b__raw_judge.jsonl` | before | 294 | 86.05% (253/294) | 4.31% (11/255) |
| `llama31_8b__qwen25_7b__raw_judge.jsonl` | before | 223 | 80.72% (180/223) | 13.29% (19/143) |
| `llama31_8b__ds_v3__raw_judge.jsonl` | after | 362 | 88.40% (320/362) | 5.21% (17/326) |
| `llama31_8b__llama4_maverick_17b__raw_judge.jsonl` | after | 323 | 86.69% (280/323) | 4.40% (12/273) |
| `llama31_8b__llama4_scout_17b__raw_judge.jsonl` | after | 327 | 81.96% (268/327) | 5.36% (14/261) |
| `llama31_8b__qwen25_72b__raw_judge.jsonl` | after | 313 | 87.54% (274/313) | 3.64% (10/275) |
| `llama31_8b__qwen25_7b__raw_judge.jsonl` | after | 232 | 79.74% (185/232) | 12.08% (18/149) |

### Position-Independent Raw Judge Mix-only (`official_mix`)

| Phase | Files | Records | Judge Accuracy | Harmful Self-Preference |
|---|---:|---:|---:|---:|
| before | 1 | 489 | 84.46% (413/489) | 6.18% (23/372) |
| after | 1 | 521 | 85.03% (443/521) | 5.25% (21/400) |

## Repro Commands

```bash
# 1) position-independent filter (normal vs swapped)
for phase in before after; do
  for normal in /Users/yidingpei/Code/Research/ms_thesis_code/data/official/judged_${phase}/*__raw_judge.jsonl; do
    swapped="${normal%.jsonl}_swapped.jsonl"
    uv run python scripts/judge_analysis/filter_position_independent.py \
      --normal-file "$normal" \
      --reverse-file "$swapped" \
      --output-file outputs/official_analysis/position_independent/${phase}/$(basename "$normal")
  done
done

# 2) metrics (same definition as analyze_self_preference.py)
uv run python scripts/judge_analysis/analyze_self_preference.py --input-file <one_file.jsonl>
```

## Position-Independent Retention

| File | Before: raw | Before: position-independent | Keep Rate | After: raw | After: position-independent | Keep Rate |
|---|---:|---:|---:|---:|---:|---:|
| `llama31_8b__ds_v3__raw_judge.jsonl` | 782 | 351 | 44.88% | 782 | 362 | 46.29% |
| `llama31_8b__llama4_maverick_17b__raw_judge.jsonl` | 699 | 307 | 43.92% | 699 | 323 | 46.21% |
| `llama31_8b__llama4_scout_17b__raw_judge.jsonl` | 655 | 299 | 45.65% | 655 | 327 | 49.92% |
| `llama31_8b__qwen25_72b__raw_judge.jsonl` | 699 | 294 | 42.06% | 699 | 313 | 44.78% |
| `llama31_8b__qwen25_7b__raw_judge.jsonl` | 655 | 223 | 34.05% | 655 | 232 | 35.42% |
| **Overall (pairwise-only)** | **3490** | **1474** | **42.23%** | **3490** | **1557** | **44.61%** |

