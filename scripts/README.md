# 脚本索引

本目录暂不按子目录移动脚本：部分运行脚本直接导入同目录的 `run_provenance.py`，保留现有路径可避免影响本地和云端命令。

## 推理编排（GPU）

- `run_mt_field_ablation.py`、`run_ccp_field_ablation.py`：既有字段删除。
- `run_mt_wrong_prompt.py`：既有错误先验。
- `run_mt_semantic_paraphrase.py`：P2-1 语义等价改写。
- `run_mt_imaging_conflict_ladder.py`：P2-2 目标成像元数据冲突。
- `run_mt_condition_field_conflicts.py`：P2-3 单字段条件冲突。
- `run_prompt_ablation.py`：提示字段消融运行器（P2 系列）。

## 本地评估与统计

- `evaluate_basic_ablation.py`：稳定的有参考基础指标。
- `evaluate_ablation_isolated.py`：逐图隔离的 NanoPyx 指标。
- `evaluate_prompt_ablation.py`：单图评估器。
- `evaluate_mt_test_nearest_baseline.py`：MT 测试图最近邻基线评估。
- `summarize_paired_ablation.py`：配对 Wilcoxon 与 Bonferroni 汇总。
- `plot_paired_ablation.py`、`plot_prompt_ablation.py`：逐图配对图。

## 诊断与协议

- `analyze_mt_metric_reliability.py`：P2-4 指标方向冲突。
- `analyze_mt_prompt_disagreement.py`：P2-5 候选提示输出分歧。
- `plot_p2_diagnostics.py`：从既有 P2-4/P2-5 CSV 生成文档诊断图；不重新推理或评估。
- `audit_biotisr_protocol.py`：BioTISR 工作簿与本地转换数据的就绪度审计。
- `audit_biosr_mt_raw_equivalence_all_levels.py`、`audit_biosr_mt_train_patch_equivalence.py`：BioSR-MT 原始 MRC 至 bundled example 的全量像素一致性审计。
- `plot_biosr_mt_preprocessing_equivalence.py`、`plot_biotisr_ccp_preprocessing_equivalence.py`：预处理实验-01/02 的图像对照生成器。
- `audit_biotisr_ccp_example_train_equivalence.py`：从原始 MRC 端到端审计 bundled example 的完整帧与实际 SR 微调补丁入口；这是 Cell_001 一致性结论的正式复现入口。
- `materialize_biosr_mt_example_from_raw.py`：将通过审计的 BioSR-MT 原始 MRC 复建为可供外部实验使用的 TIFF；默认拒绝覆盖。
- `materialize_biosr_mt_verified_assets.py`：阶段 1–2 的 BioSR-MT 生产器；先冻结整个 bundled 目录基线，再在新的 `data/derived/` 目录中严格生产已审计的全图、训练 patch 与索引，默认拒绝覆盖。
- `audit_biosr_mt_fluoresfm_inference.py`：阶段 3 的无覆盖上游推理探针；审计 `*_fluoresfm` 测试图是否可由已保存的模型、提示和 napari 推理实现重新生成。
- `probe_biosr_mt_fluoresfm_server.py`：阶段 3 的服务器单图探针；冻结服务器环境与 napari-fluoresfm 源码/checkpoint 身份，隔离单张输入运行固定版本上游推理，逐文件判定 strict / 数值等价 / 失败，默认拒绝覆盖。
- `sync_probe_to_server.py`：把阶段 3 探针资产（脚本、配置与 napari-fluoresfm 子模块）一键推送到云端个人目录；优先 WSL rsync、回退 scp；`--include-probe-data` 传最小探针数据，`--include-data` 传完整 example；凭据不入库。
- `produce_biosr_mt_fluoresfm_batch.py`：阶段 3.3/4 批量生产器；用冻结生产配置（patch=64 + 去卷积 prompt）生产 90 张 `*_fluoresfm` 全图 + 735 测试 patch，逐文件按官方口径验证（质量等价）；支持 `--evaluate-only`（复用输出重分类）与 `deterministic: true` 的可选跨进程确定性变体（运行时注入 benchmark=False，磁盘源码不变）；拒绝覆盖。
- `diag_numeric_variation.py`：确定性诊断；跨进程/同进程逐字节复现实验，`--benchmark-off`（运行时注入，不落盘）、`--double`（同进程两次）、`--deterministic`（强制确定性算法）。
- `eval_biosr_mt_official_metrics.py`：官方口径（每图 P3/P99.5 → clip[0,2.5] → data_range=2.5）复核；含背景扣除、8/15 图、SIM 降采样方法对比。
- `eval_biosr_mt_batch_vs_sim.py`：SIM 锚定等价验证；90 图生产候选 vs SIM 与官方参考 vs SIM 对比（CPU-only）。
- `final_acceptance_biosr_mt.py`：全目录最终验收汇总；合并阶段 1–4 证据（基线清单 + 阶段 2 验证 + 阶段 3–4 批量 comparison），给全部 83,591 文件唯一状态并核验无遗漏/重复/排除。
- `plot_biosr_mt_inference_evidence.py`：推理实验-01..04 的图形证据生成器（patch 设置质量对比、逐图指标、真实图像对照、全级别/SIM 锚定、最终验收状态覆盖）；读取 manifest/CSV。
- `run_provenance.py`：运行清单、资产哈希和 `run.md` 写入工具。

## 提交前配置检查

运行 `python scripts/check_config_path_policy.py`。它要求 `configs/` 的键为 ASCII `lower_snake_case`，并要求项目自建输出路径为不含空格和非 ASCII 字符的相对路径；上游原始数据路径不受此限制。

详细实验状态见 [文档索引](../docs/README.md) 与 [当前进度](../docs/progress/进度_当前总览.md)。
