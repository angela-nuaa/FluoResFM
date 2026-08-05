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
- `plot_p2_diagnostics.py`：从既有 P2-4/P2-5 CSV 生成文档诊断图；不重新推理或评估。
- `audit_biotisr_protocol.py`：BioTISR 工作簿与本地转换数据的就绪度审计。
- `audit_biosr_mt_raw_equivalence_all_levels.py`、`audit_biosr_mt_train_patch_equivalence.py`：BioSR-MT 原始 MRC 至 bundled example 的全量像素一致性审计。
- `audit_biotisr_ccp_example_train_equivalence.py`：从原始 MRC 端到端审计 bundled example 的完整帧与实际 SR 微调补丁入口；这是 Cell_001 一致性结论的正式复现入口。
- `materialize_biosr_mt_example_from_raw.py`：将通过审计的 BioSR-MT 原始 MRC 复建为可供外部实验使用的 TIFF；默认拒绝覆盖。
- `run_provenance.py`：运行清单、资产哈希和 `run.md` 写入工具。

## 提交前配置检查

运行 `python scripts/check_config_path_policy.py`。它要求 `configs/` 的键为 ASCII `lower_snake_case`，并要求项目自建输出路径为不含空格和非 ASCII 字符的相对路径；上游原始数据路径不受此限制。

详细实验状态见 [文档索引](../docs/README.md) 与 [当前进度](../docs/progress/进度_当前总览.md)。
