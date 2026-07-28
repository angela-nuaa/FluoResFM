# FluoResFM

> FluoResFM 的个人学习、复现与提示条件实验记录；非官方仓库。

原论文提出以任务、样本结构和成像条件为文本先验的荧光显微图像恢复模型，涵盖去噪、去卷积和超分辨率。本仓库保留可复现环境、上游代码引用与经核对的本地实验脚本；不发布原始数据、权重或 TIFF 结果。

- 原论文：[Lu et al., *Nature Communications* (2026)](https://doi.org/10.1038/s41467-026-70307-4)
- 上游实现：[qiqi-lu/fluoresfm](https://github.com/qiqi-lu/fluoresfm)
- 本仓库的范围、结果与限制：[实验结果与边界](docs/实验结果与边界.md)

## 快速开始

在 WSL2 或 Linux 中：

```bash
conda env create -f environment.yml
conda activate fluoresfm
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

cd repos/fluoresfm
python 0_7_test_model.py
```

预训练权重和示例数据不随 Git 分发。下载来源、目录和许可见 [DATA_AND_MODELS.md](DATA_AND_MODELS.md)。

交互式推理可使用 napari：

```bash
pip install napari-fluoresfm
napari
```

随后在 napari 中选择 `Plugins → napari-fluoresfm`。训练、命令行推理及上游评估脚本位于子模块 `repos/fluoresfm/`。

## 本仓库做了什么

所有正式记录均使用随附的规范提示文本：BioSR_MT 使用 `example/data/text/train/dataset_text_ALL.txt`，BioTISR_CCP 使用 `example/data/text/finetune/dataset_text_ALL.txt`。工作簿支持两者均为 2× 超分辨率映射，但并不提供当前 MT 测试图的逐文件元数据；因此以下是提示敏感性实验，不是论文官方测试协议或性能复现。

| 实验 | 数据范围 | 已观察到的结果 | 不能推出的结论 |
|---|---|---|---|
| MT 字段删除 | 15 张 BioSR_MT 捆绑测试图 | 去除成像、任务、结构字段均降低有参考图保真度；成像字段的影响最大。 | 不能证明对所有显微镜、样本或任务均成立。 |
| MT 错误先验 | 同一 15 张图 | 错误任务的损失最大；错误结构较小；仅将目标模态改为宽场的影响很小。 | 不能据此认定模型理解了全部成像语义。 |
| CCP 字段删除 | 20 张 BioTISR_CCP 捆绑训练目录样本 | 去除任务字段的损失最大；SSIM 与其他指标出现不一致。 | 不是独立测试集，不是泛化性能。 |

完整均值、配对检验、提示构造和限制见 [实验结果与边界](docs/实验结果与边界.md)。早期探索性运行保留在本地忽略目录 `experiments/`，不作为仓库主结论。

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

## 指标解释

PSNR、SSIM、MS-SSIM、ZNCC、RSP、NRMSE 与 RSE 均以对应 SIM 参考图评价保真度。DA 分辨率是不依赖参考图的诊断量；它可与有参考指标给出相反排序，不能单独作为“重建更好”的证据。展示图采用独立百分位归一化，仅用于定性检查，不能比较绝对强度。

## 项目结构

```text
repos/                 上游 FluoResFM 与 napari 插件子模块
scripts/               本仓库的提示实验、评估与统计脚本
docs/                  论文阅读材料与实验研究记录
example/               本地下载的上游示例资源（忽略 Git）
experiments/           本地原始预测、指标和日志（忽略 Git）
environment.yml        已验证的环境定义
```

## 文档与引用

- [论文速读](docs/FluoResFM_论文速读.md) · [论文精读](docs/FluoResFM_论文精读.md) · [完整翻译](docs/FluoResFM_完整翻译.md)
- [数据、模型与许可](DATA_AND_MODELS.md) · [变更记录](CHANGELOG.md)

研究结论、方法或模型请引用原论文，而非本仓库：

```bibtex
@article{lu2026fluoresfm,
  title={A foundation model for multi-task cross-distribution restoration of fluorescence microscopy images},
  author={Lu, Qiqi and Liu, Xiuli and Feng, Qianjin and Zeng, Shaoqun and Cheng, Shenghua},
  journal={Nature Communications}, volume={17}, pages={3729}, year={2026},
  doi={10.1038/s41467-026-70307-4}
}
```
