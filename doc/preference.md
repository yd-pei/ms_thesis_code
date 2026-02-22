# Self Preference

## Results

=== Per-file metrics ===

[llama31_70.jsonl]
  self_label_field=llama31_70b_output_label
  total_records=244
  model_accuracy=20.49% (50/244)
  self_selected_rate=18.85% (46/244)
  harmful_self_preference_rate=67.39% (31/46)

[llama31_8.jsonl]
  self_label_field=llama31_8b_output_label
  total_records=192
  model_accuracy=7.29% (14/192)
  self_selected_rate=7.81% (15/192)
  harmful_self_preference_rate=93.33% (14/15)

[llama33_70.jsonl]
  self_label_field=llama33_70b_output_label
  total_records=233
  model_accuracy=19.74% (46/233)
  self_selected_rate=30.90% (72/233)
  harmful_self_preference_rate=70.83% (51/72)

[qwen25_7.jsonl]
  self_label_field=qwen25_7b_output_label
  total_records=205
  model_accuracy=12.68% (26/205)
  self_selected_rate=14.15% (29/205)
  harmful_self_preference_rate=89.66% (26/29)