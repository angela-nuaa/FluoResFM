# FluoResFM

> FluoResFM 的个人学习、复现与提示条件实验记录；非官方仓库。

原论文提出以任务、样本结构和成像条件为文本先验的荧光显微图像恢复模型，涵盖去噪、去卷积和超分辨率。本仓库保留可复现环境、上游代码引用与经核对的本地实验脚本；不发布原始数据、权重或 TIFF 结果。

- 原论文：[Lu et al., *Nature Communications* (2026)](https://doi.org/10.1038/s41467-026-70307-4)
- 上游实现：[qiqi-lu/fluoresfm](https://github.com/qiqi-lu/fluoresfm)
- 文档入口与当前状态：[文档索引](docs/README.md) · [当前进度](docs/progress/进度_当前总览.md) · [可复现性与验收边界](docs/REPRODUCIBILITY.md)

## 术语

- **bundled example**：napari 插件自带的上游示例数据（本仓库复现基准）。
- **`_fluoresfm` 参考**：作者 FluoResFM 模型在示例上的推理输出（复现目标）。
- **官方口径**：论文评估方法——每图 P3/P99.5（第 3/99.5 百分位）归一化 → clip[0,2.5] → PSNR/SSIM `data_range=2.5`。
- **质量等价**：生产判定标准（非逐字节；path/shape/dtype 一致 + 官方口径指标阈值）。

## 目前状态

已完成两项 bundled example 的预处理审计，并完成 BioSR-MT 的 FluoResFM 推理生产与全目录验收。

预处理：

| 实验 | 已验证范围 | 结论 |
| --- | --- | --- |
| [预处理实验-01](docs/experiment_results/预处理实验-01_BioSR-MT原始MRC与example一致性审计.md) | BioSR-MT 测试、训练全图及实际训练 patch | 135 张测试 WF、15 张测试 SIM、360 张训练 WF 严格相等；147,960 个 patch 在 float32 容差内一致。 |
| [预处理实验-02](docs/experiment_results/预处理实验-02_BioTISR-CCP原始MRC与example一致性审计.md) | BioTISR-CCP bundled example 的 `Cell_001` | 60 张 WF、20 张 SIM 严格相等；256 对 SR 补丁在 float32 容差内一致。 |

推理生产与验收（判定标准为质量等价，非逐字节；详见 [推理实验-01](docs/experiment_results/推理实验-01_BioSR-MT_FluoResFM参数探索与生产决策.md) 至 [推理实验-04](docs/experiment_results/推理实验-04_全目录最终验收审计报告.md)）：

| 项 | 结论 |
| --- | --- |
| 生产配置 | patch=64、overlap=16、batch=8、compile=false、sf_lr=1、无 flash；15 图近精确复现 bundled 参考（官方口径 48.13 dB / 0.9974）。 |
| 官方实现对比 | 主链路与官方 `3_0_test_it2i.py` 逐行等价；生产 prompt ≡ 官方元数据；评估口径校正（每图 P3/P99.5 → clip[0,2.5] → data_range=2.5，无背景扣除）。 |
| 批量生产 | 90 张 `*_fluoresfm` 全图 + 735 测试 patch（质量等价）；SIM 锚定等价（90 图全级别：候选 vs SIM 34.68 ≈ 官方参考 34.63 dB）。 |
| 全目录验收 | **83,591 文件唯一状态**：566 严格 + 82,200 数值等价 + 825 质量等价，无遗漏/重复/排除。 |
| 确定性 | 同进程内逐字节确定、跨进程/跨 GPU 有低阶位差；可选确定性变体可跨进程逐字节复现（磁盘源码不变）。 |

可复用规范见 `docs/protocol/`（预处理、提示词编写、推理生产与验收三份协议）。其中 566「严格」= 预处理字节一致资产（MRC→example），825「质量等价」= 推理输出（`_fluoresfm` 全图 + 测试 patch）。以上均为 bundled example 的内容级证据，不等价于论文完整训练/测试划分；其余数据的外推须逐文件审计。

## 快速开始

在 WSL2 或 Linux 中（CUDA 12.4）：

```bash
git clone --recurse-submodules <repository-url> FluoResFM
cd FluoResFM
git submodule update --init --recursive
conda env create -f environment.yml
conda activate fluoresfm
pip install -e repos/napari-fluoresfm --no-deps
python scripts/doctor.py
```

这一步只建立并诊断运行环境，不下载任何资产。预训练权重和示例数据不随 Git
分发；下载、归档 SHA-256 与解压后路径校验见 [DATA_AND_MODELS.md](DATA_AND_MODELS.md)
和 [可复现性与验收边界](docs/REPRODUCIBILITY.md)。资产准备好后执行
`python scripts/doctor.py --assets --production`，再运行相应生产命令。
环境包含 `PyQt5`：虽批量入口不打开 GUI，但冻结的 napari 插件在导入预测器时会
加载 Qt widget，缺少 Qt 后端会导致命令行生产在启动时失败。

交互式推理可使用 napari：

```bash
napari
```

随后在 napari 中选择 `Plugins → napari-fluoresfm`。上述开发安装固定使用仓库子模块；
若只想体验上游发布版，可另建环境后执行 `pip install napari-fluoresfm`，但它不保证
与本仓库的冻结生产配置一致。训练、命令行推理及上游评估脚本位于子模块 `repos/fluoresfm/`。

## 提示敏感性实验（P2 线）

这是独立于「推理生产」（见上）的另一条历史实验线。所有正式记录均使用随附的规范提示文本：BioSR_MT 使用 `example/data/text/train/dataset_text_ALL.txt`，BioTISR_CCP 使用 `example/data/text/finetune/dataset_text_ALL.txt`。工作簿支持两者均为 2× 超分辨率映射，但并不提供当前 MT 测试图的逐文件元数据；因此以下是提示敏感性实验，不是论文官方测试协议或性能复现。

| 实验 | 数据范围 | 已观察到的结果 | 不能推出的结论 |
|---|---|---|---|
| MT 字段删除 | 15 张 BioSR_MT 捆绑测试图 | 去除成像、任务、结构字段均降低有参考图保真度；成像字段的影响最大。 | 不能证明对所有显微镜、样本或任务均成立。 |
| MT 错误先验 | 同一 15 张图 | 错误任务的损失最大；错误结构较小；仅将目标模态改为宽场的影响很小。 | 不能据此认定模型理解了全部成像语义。 |
| CCP 字段删除 | 20 张 BioTISR_CCP 捆绑训练目录样本 | 去除任务字段的损失最大；SSIM 与其他指标出现不一致。 | 不是独立测试集，不是泛化性能。 |

早期探索的均值、配对检验、提示构造和限制见 [早期提示探索实验（历史归档）](docs/experiment_results/历史_提示探索实验_结果与边界.md)。早期探索性运行保留在本地忽略目录 `experiments/`，不作为仓库主结论。

## 复运行规范实验

GPU 仅用于推理；指标计算、配对统计和作图在本地进行。每次运行使用新目录，保留 `manifest.json`，不要覆盖既有结果。

```bash
# MT 字段删除：云端或本地 GPU 推理
python scripts/run_mt_field_ablation.py --protocol wf_to_sim_2x --device cuda:0 \
  --runs-dir experiments/mt_field_ablation/<run-id>

# CCP 字段删除：云端或本地 GPU 推理
python scripts/run_ccp_field_ablation.py --device cuda:0 \
  --runs-dir experiments/ccp_field_ablation/<run-id>

# 本地评估（将 BioSR_MT 替换为 BioTISR_CCP 即可）
python scripts/evaluate_basic_ablation.py --dataset BioSR_MT --runs-dir <run-dir>
python scripts/evaluate_ablation_isolated.py --dataset BioSR_MT --runs-dir <run-dir> --workers 8
python scripts/summarize_paired_ablation.py \
  --metrics <run-dir>/isolated_full_metrics.csv \
  --output <run-dir>/full_paired_summary.csv
```

`evaluate_ablation_isolated.py` 按图像隔离运行并支持断点续跑，以避免 NanoPyx 批量处理的资源累积。统计前应核对每个条件的图像数和文件名集合。

## 复运行推理生产（BioSR-MT，质量等价）

> 生产在作者的个人云端 GPU 环境运行；云端设置（连接、同步、环境）不随仓库公开。此处给出可复现的脚本入口。

该入口需要本地 CUDA GPU、已校验的下载资产和两个固定子模块；它不会让没有资产的
Git checkout 自行重算既有结论。已提交的机器可读结论索引见
[`results/biosr-mt-production-v1.summary.json`](results/biosr-mt-production-v1.summary.json)。

生产协议、参数与验收标准见 [`docs/protocol/协议_BioSR-MT推理生产与验收.md`](docs/protocol/协议_BioSR-MT推理生产与验收.md)。批量生产（90 全图 + 735 测试 patch）入口：

```bash
python scripts/produce_biosr_mt_fluoresfm_batch.py \
  --config configs/biosr_mt_fluoresfm_batch_production_v1.json \
  --workspace-root <FluoResFM 仓库根目录>
```

判定为质量等价（非逐字节）；跨进程逐字节复现可用确定性变体配置（`..._deterministic_v1.json`）。

## 指标解释

PSNR、SSIM、MS-SSIM、ZNCC、RSP、NRMSE 与 RSE 均以对应 SIM 参考图评价保真度。DA 分辨率是不依赖参考图的诊断量；它可与有参考指标给出相反排序，不能单独作为“重建更好”的证据。展示图采用独立百分位归一化，仅用于定性检查，不能比较绝对强度。

## 项目结构

```text
repos/                 上游 FluoResFM 与 napari 插件子模块
scripts/               本仓库的提示实验、评估与统计脚本
docs/                  文档索引、进度、计划、结果、协议与论文笔记
example/               本地下载的上游示例资源（忽略 Git）
experiments/           本地原始预测、指标和日志（忽略 Git）
environment.yml        已验证的环境定义
requirements/          直接依赖约束与可选加速依赖
assets/                本地下载资产的可审阅清单模板
results/               机器可读的结论索引（不含原始 TIFF/逐文件 CSV）
```

## 文档与引用

- [论文速读](docs/literature/文献_FluoResFM论文速读.md) · [论文精读](docs/literature/文献_FluoResFM论文精读.md) · [完整翻译](docs/literature/文献_FluoResFM完整翻译.md)
- [脚本索引](scripts/README.md) · [数据、模型与许可](DATA_AND_MODELS.md) · [变更记录](CHANGELOG.md)

研究结论、方法或模型请引用原论文，而非本仓库：

```bibtex
@article{lu2026fluoresfm,
  title={A foundation model for multi-task cross-distribution restoration of fluorescence microscopy images},
  author={Lu, Qiqi and Liu, Xiuli and Feng, Qianjin and Zeng, Shaoqun and Cheng, Shenghua},
  journal={Nature Communications}, volume={17}, pages={3729}, year={2026},
  doi={10.1038/s41467-026-70307-4}
}
```
