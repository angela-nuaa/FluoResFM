# 脚本索引

本目录暂不按子目录移动脚本：部分运行脚本直接导入同目录的 `run_provenance.py`，保留现有路径可避免影响本地和云端命令。

## 推理编排（GPU）

- `run_mt_field_ablation.py`、`run_ccp_field_ablation.py`：既有字段删除。
- `run_mt_wrong_prompt.py`：既有错误先验。
- `run_mt_semantic_paraphrase.py`：P2-1 语义等价改写。
- `run_mt_imaging_conflict_ladder.py`：P2-2 目标成像元数据冲突。
- `run_mt_condition_field_conflicts.py`：P2-3 单字段条件冲突。

## 本地评估与统计

- `evaluate_basic_ablation.py`：稳定的有参考基础指标。
- `evaluate_ablation_isolated.py`：逐图隔离的 NanoPyx 指标。
- `evaluate_prompt_ablation.py`：单图评估器。
- `summarize_paired_ablation.py`：配对 Wilcoxon 与 Bonferroni 汇总。
- `plot_paired_ablation.py`、`plot_prompt_ablation.py`：逐图配对图。

## 诊断与协议

- `analyze_mt_metric_reliability.py`：P2-4 指标方向冲突。
- `analyze_mt_prompt_disagreement.py`：P2-5 候选提示输出分歧。
- `audit_biotisr_protocol.py`：BioTISR 工作簿与本地转换数据的就绪度审计。
- `run_provenance.py`：运行清单、资产哈希和 `run.md` 写入工具。

详细实验状态见 [文档索引](../docs/README.md) 与 [当前进度](../docs/progress/当前进度总览.md)。
