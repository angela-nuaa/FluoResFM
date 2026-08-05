# BioTISR 外部 SR 协议就绪度审计

> 状态：外部 SR 评测仍未就绪；本地 bundled example 的预处理已单独验证。更新：2026-08-05。

## 结论

上游 `dataset_test-v2.xlsx` 定义了 12 个 BioTISR 外部超分辨率条目（4 类结构 × 3 个 level），但本地 `data/raw/BioTISR/` 未包含其所需的输入 TIFF、参考图 TIFF、图像索引或逐文件映射。

| 检查项 | 就绪数 |
| --- | ---: |
| 规范 SR 条目 | 12 |
| 索引、输入、参考图均存在 | 0 |
| 可逐文件配对评测 | 0 |

因此，当前不能将本地 RawSIM 直接当作工作簿定义的外部测试集，也不能报告 P0/P1 级别的零样本评测。

## 与预处理实验-02的关系

[预处理实验-02](../experiment_results/预处理实验-02_BioTISR-CCP原始MRC与example一致性审计.md) 已直接证明：`Cell_001` 原始 MRC 可精确复现 bundled example 的三种 WF 水平、SIM 和 SR 微调补丁入口。

这不补足外部 SR 工作簿的细胞/时间点映射、索引或测试划分；两项结论并不冲突。

## 恢复严格复现所需材料

以下任一材料到位后，先运行 `scripts/audit_biotisr_protocol.py`：

1. 作者已转换的输入、参考图和索引；或
2. 可审计的 RawSIM→WF/GT 流程，包含参数、逐文件映射和测试划分。

在此之前，所有结果只能称为 bundled example 的内容级预处理验证，不能称为论文外部测试或近似论文性能复现。
