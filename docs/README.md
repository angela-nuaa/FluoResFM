# 文档索引

## 先看这里

- [当前进度](progress/进度_当前总览.md)：完成项、协议阻塞与恢复工作的顺序。
- [预处理与数据复现计划](plans/计划_预处理与数据复现.md)：预处理审计、数据依赖与 P0/P1/P2 边界。

## 结果

- [早期提示探索实验与边界（历史归档）](experiment_results/历史_提示探索实验_结果与边界.md)
- [P2-1：语义等价改写](experiment_results/P2-1_语义等价改写结果.md)
- [P2-2：错误目标元数据冲突](experiment_results/P2-2_错误元数据冲突阶梯结果.md)
- [预处理实验-01：BioSR-MT 原始 MRC 与 bundled example 一致性审计](experiment_results/预处理实验-01_BioSR-MT原始MRC与example一致性审计.md)
- [预处理实验-02：BioTISR-CCP 原始 MRC 与 bundled example 一致性审计](experiment_results/预处理实验-02_BioTISR-CCP原始MRC与example一致性审计.md)
- [P2-3：单字段条件冲突](experiment_results/P2-3_单字段条件冲突结果.md)
- [P2-4：指标可靠性](experiment_results/P2-4_指标可靠性诊断结果.md)
- [P2-5：提示分歧风险信号](experiment_results/P2-5_提示分歧风险信号结果.md)

## 协议与数据

- [BioTISR 协议就绪度审计](protocol/协议_BioTISR就绪度审计.md)

## 文献笔记

- [论文速读](literature/文献_FluoResFM论文速读.md)
- [论文精读](literature/文献_FluoResFM论文精读.md)
- [完整翻译](literature/文献_FluoResFM完整翻译.md)

## 命名规则

- `experiment_results/`：`预处理实验-编号_...`、`P2-编号_...` 或 `历史_...`。
- 其余子目录：以 `计划_`、`进度_`、`协议_`、`文献_` 标明文档类型。
- 文档标题、正文、图题和 `docs/` 内文件名可使用中文；运行日志也使用中文。
- 配置键使用 ASCII `lower_snake_case`。项目自建的运行输出路径仅使用 ASCII，并以稳定的 `experiment_id` 命名；例如 `preprocess-01_biosr-mt-raw-equivalence`。中文说明写入 `experiment_label`。
- 上游原始数据的官方目录名可保留原样（包括空格）；脚本必须通过 `pathlib` 或等价的正确引用方式访问，禁止以重命名原始数据为代价满足本约定。
- 提交前运行 `python scripts/check_config_path_policy.py`，检查配置键及项目自建输出路径。
- [FluoResFM PDF](literature/FluoResFM.pdf)

原始数据、权重、预测 TIFF、日志和逐图指标均不进入此目录；它们保存于 Git 忽略的 `data/`、`example/` 与 `experiments/`。
