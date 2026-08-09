# 数据、模型与实验产物

本仓库只跟踪可审阅的代码、配置与文档；不提交原始数据、预训练权重、缓存或实验输出。这样可避免大文件、受限再分发与不可复现的二进制产物进入版本历史。

| 资源 | 本地目录 | 权威来源 | 处理规则 |
| --- | --- | --- | --- |
| 示例训练/测试数据 | `example/data/` | [Zenodo 18382702](https://doi.org/10.5281/zenodo.18382702) | 下载后将来源记录中的档案名、档案 SHA-256 和本地路径写入 `assets/manifest.local.json`；不提交。 |
| FluoResFM 与 BiomedCLIP 权重 | `example/checkpoints/` | [Zenodo 18382702](https://doi.org/10.5281/zenodo.18382702) | 同上；解压后由 `python scripts/doctor.py --assets --production` 校验路径。 |
| 自建或派生数据 | `data/` | 由使用者记录 | 保留不可变原始副本；将清洗/切片结果写到独立派生目录，并记录脚本、参数和随机种子。 |
| 实验产物 | `experiments/` 或 `outputs/` | 本仓库运行生成 | 每次运行记录 Git 提交、子模块提交、环境、硬件、命令、随机种子和输入数据版本。 |

上游论文说明：研究使用的数据来自既有公开文献；示例数据、预训练编码器和模型由 Zenodo 提供；其他大规模原始及中间数据需向通讯作者请求。请遵守各数据集、模型和上游仓库的许可证与使用条款。

## 下载与落位

仓库不代替来源站点重新分发资产，也不在文档中臆造档案名或哈希。请从来源记录取得
它们后，复制 `assets/manifest.local.example.json` 为 `assets/manifest.local.json`，填写
`archive_filename`、`archive_path` 和 `sha256`；`archive_path` 相对于仓库根目录。
然后运行 `python scripts/doctor.py --assets --production`。前者验证原始下载档案，后者
验证当前批量生产所需的解压目标路径。完整工作流见
[可复现性与验收边界](docs/REPRODUCIBILITY.md)。

## 最小实验记录

在每个实验输出目录创建 `run.md`，至少记录：目标、执行命令、根仓库及两个子模块的提交 ID、`conda env export --from-history` 输出、GPU/CUDA/PyTorch 版本、随机种子、输入文件清单及 SHA-256、评价脚本与指标。未经明确批准，不覆盖原始输入或既有结果。
