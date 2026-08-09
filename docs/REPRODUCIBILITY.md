# 可复现性与验收边界

本仓库可复现的是已明确写入配置、脚本和文档的流程；不因仓库存在一份结果
报告，就自动等价于论文完整训练/测试协议的复现。当前已报告结论的范围必须以
每份实验报告和 `results/*.summary.json` 中的 `scope` 为准。

## 从零安装

在 WSL2 或 Linux（CUDA 12.4 对应 PyTorch wheel）中执行：

```bash
git clone --recurse-submodules <repository-url> FluoResFM
cd FluoResFM
git submodule update --init --recursive
conda env create -f environment.yml
conda activate fluoresfm
pip install -e repos/napari-fluoresfm --no-deps
python scripts/doctor.py
```

`doctor` 同时检查两个固定的子模块提交和本仓库脚本直接依赖。它不下载数据，
也不运行 GPU 推理。若要使用 FlashAttention，必须在已通过 `doctor` 的 Linux
环境中按 `requirements/optional-linux.txt` 单独安装；当前 BioSR-MT 生产配置
并不使用它。

## 数据与权重

先按 [数据、模型与实验产物](../DATA_AND_MODELS.md) 的权威来源取得文件。将
`assets/manifest.local.example.json` 复制为 `assets/manifest.local.json`，从来源
记录填写档案名、档案本地路径和 SHA-256；不要猜测哈希或以解压后的文件替代原始
档案哈希。

```bash
python scripts/doctor.py --assets --production
```

其中 `--assets` 校验下载档案本身，`--production` 额外检查当前 BioSR-MT 批量
生产器所需的已解压路径。二者都不写入资产。

## 锁定与升级

`requirements/runtime-cu124-py312.txt` 是在本地已运行的 Python 3.12 / CUDA
12.4 组合中记录的**直接依赖约束**，不是跨操作系统的完整锁文件。GPU、驱动、
平台相关 wheel（尤其是 Triton/FlashAttention）不能诚实地由一个未测试的通用
文件锁定。

升级任何依赖或子模块后，应当：

1. 新建输出目录，运行最小 smoke test 和目标生产配置；
2. 保存 `pip freeze`、GPU/CUDA 信息、输入 SHA-256 与生成的 manifest；
3. 用独立保留样本确定或复核验收阈值；
4. 仅在结果通过后更新约束、配置 hash 和 `results/*.summary.json`。

## 已提交证据与未提交产物

`results/biosr-mt-production-v1.summary.json` 是机器可读的结果索引，明确标记
其是否能从当前 Git 树独立重算。原始 TIFF、权重和逐文件 CSV 不提交，原因是
体积、许可与再分发限制；因此阅读者应把当前报告视为可追溯的主张，而不是可在
无资产条件下独立复算的原始证据。
