# BioSR-MT bundled example 全目录严格生产计划

> 状态：**已全部完成（2026-08-08）**。阶段 1–2 已完成；阶段 3–4（90 `_fluoresfm` 全图 + 735 测试 patch）与最终验收（83,591 文件唯一状态 + 中文审计报告）已完成，结论为「严格逐字节不可行、质量等价达成」，判定标准已按用户指示放宽为质量等价。审计报告见 [`推理实验-04_全目录最终验收审计报告.md`](../experiment_results/推理实验-04_全目录最终验收审计报告.md)。
>
> 目标：在不改写 `example/data/BioSR_MT` 的前提下，生产该 bundled example 目录中全部受审计文件，并对每个文件保留可复查的来源、配置、环境与逐文件验证证据。

## 范围与结论边界

- 本计划的对象是 bundled example 的文件级生产，不是 FluoResFM 论文的完整复现，也不构成对论文数据划分、训练过程或性能结论的证明。
- 所有候选输出写入新的 Git 忽略运行目录；`example/data/BioSR_MT` 仅作不可变参考，不覆盖、不清理、不就地生成。
- “生产”的判定标准（阶段 3 后已放宽）：参考文件与候选文件的路径、shape、dtype 必须一致；数值层面按 **质量等价** 判定——候选 vs bundled 参考的 PSNR/SSIM 不低于阶段 3 已验证基线（官方口径，15 图 patch=64：PSNR 48.13 dB、SSIM 0.9974，最低 46.0 dB）。不得以 SHA-256 一致作为生产判定（2026-08-08 修订：同进程内逐字节确定、**跨进程/跨 GPU 有低阶位差**，无法跨进程复现他人字节；详见 [`推理实验-03`](../experiment_results/推理实验-03_官方推理实现对比与待确认项.md) §5）。
- 阶段 3 推理结论见 [`推理实验-01_BioSR-MT_FluoResFM推理输出审计.md`](../experiment_results/推理实验-01_BioSR-MT_FluoResFM推理输出审计.md) 与 [`推理实验-03_官方推理实现对比与待确认项.md`](../experiment_results/推理实验-03_官方推理实现对比与待确认项.md)；生产配置见 `configs/biosr_mt_fluoresfm_production_v1.json`。云端运行环境与同步速查见下文「云端运行环境与同步」节。
- 每次运行必须记录输入 SHA-256、脚本与配置版本、Git commit、Python/Torch/CUDA/驱动版本、GPU 型号、随机种子（如适用）、命令、stdout/stderr 与逐文件比较结果。

## 参考基线与当前覆盖

阶段 1 已冻结 `example/data/BioSR_MT` 的完整基线：83,591 个参考文件（83,575 个 TIFF、16 个文本文件），以及 550 个公开原始 MRC 输入的清单。

阶段 2 已完成以下已验证资产的独立生产，运行目录为 `data/derived/biosr-mt-example-verified-assets/20260805_r3/`：

| 资产 | 数量 | 验证规则 | 状态 |
| --- | ---: | --- | --- |
| 测试与训练全图 WF/SIM | 550 TIFF | 最大绝对误差为 0 | 已完成 |
| 训练 patch | 82,200 TIFF | float32 容差 `1e-6`；最大误差 `2.384185791015625e-07` | 已完成（数值等价） |
| 根目录索引与嵌套元数据文本 | 16 文件 | 精确字节比较 | 已完成 |

阶段 2 总计生成 82,750 个 TIFF 与 16 个文本文件。其入口为：

- `configs/biosr_mt_verified_materialization_20260805_r3.json`
- `scripts/materialize_biosr_mt_verified_assets.py`

## 仍待生产的资产

| 资产族 | 数量 | 当前证据 | 后续要求 |
| --- | ---: | --- | --- |
| `test/channel_0/WF_noise_level_{1,2,3,4,5,7}_fluoresfm/*.tif` | 90 | 目录命名与 upstream `predict.py` 表明这是 FluoResFM 推理输出，不是原始 MRC 的纯预处理结果 | 在服务器锁定历史推理环境、checkpoint、文本提示词及推理参数后，先单图验证，再批量生产 |
| `test/channel_0/WF_noise_level_3_fluoresfm_p64_s64_2d/*.tif` | 735 | 已发现候选链：对 `_fluoresfm` 全图非负截断后做 P3/P99.5 归一化，再以 64×64、stride 64 裁切；现有数值差最大约 `2.28e-7` | 只有对应 `_fluoresfm` 全图通过阶段 3 后，才可将该链纳入严格生产与全量验证 |

## 阶段 3：`_fluoresfm` 推理全图逆向审计与生产

### 3.1 冻结服务器运行条件

1. 使用受限的普通工作账户运行；禁止 sudo、禁止写共享数据、checkpoint 与其他用户目录。
2. 工作区、缓存、临时文件与候选输出均位于该账户的个人目录；原始输入与 checkpoint 使用只读路径。
3. 记录服务器环境：GPU、驱动、CUDA、cuDNN、Python、Torch、Triton、`pip freeze` 或等价锁定清单。
4. 锁定 napari-fluoresfm 历史源码与 checkpoint 的 SHA-256。历史候选源码提交包括 `d62340321d58b27cb7020ff011ed6126deb49425`；必须以实际验证结果确定最终版本。

### 3.2 单图探针（先于任何批量运行）

1. 选定一个参考全图，例如 `WF_noise_level_3` 的 `41.tif`，建立只含该输入的候选运行目录。
2. 固定以下变量并逐项记录：输入文件、checkpoint、prompt、`sf_lr`、`batch_size`、`patch_size`、AMP、`torch.compile`、模型源码 commit 与依赖版本。
3. 历史推理参数经阶段 3 实证修正为 `patch_size=64`：bundled `_fluoresfm` 参考在 patch=64 下可近精确复现（15 图 48.13 dB，官方口径），显著优于 patch=256（42.00 dB）；patch=64 亦与代码/GUI 默认一致。`sf_lr=1`、去卷积 prompt 与 GUI 默认 prompt。
4. 将候选 TIFF 与参考 TIFF 比较 shape、dtype、数组差异与 SHA-256。只有满足阶段开头的严格规则，才进入批量生产。
5. 本机候选探针不能作为结论：当前本机运行 `torch.compile` 时因 Torch/Triton 不兼容而失败；服务器须记录是否采用兼容环境。关闭 compile 的结果仅用于隔离变量，不得默认等价于历史输出。

### 3.3 批量生产与验证

1. 单图通过后，对 6 个噪声级别的 90 张全图在新的运行目录中生产。
2. 每级别完成后立即逐文件验证；任一文件不满足严格规则即停止，不覆盖参考，也不将其余文件标为成功。
3. 产出机器可读的 `manifest.json`、输入/输出索引、环境锁定文件、逐文件 `comparison.csv`、运行日志与失败原因。
4. 将成功的生成规则固化为受版本控制的配置与脚本；运行产物保持 Git 忽略。

## 阶段 4：测试 patch 严格生产

1. 以阶段 3 严格通过的 level-3 `_fluoresfm` 全图为唯一输入。
2. 对每张全图执行已发现的候选转换：非负截断、P3/P99.5 归一化、64×64 patch、stride 64；实现必须显式定义百分位计算、边界处理、dtype 与 TIFF 写入参数。
3. 生产全部 735 个 patch，逐文件与 bundled example 比较路径、shape、dtype、数组内容和 SHA-256。
4. 如出现仅浮点容差内一致而文件哈希不一致，记录数值等价与差异来源；不标为文件级严格一致。

## 云端运行环境与同步（批量生产用）

- 连接：`ssh lc-Dual-4090`（`~/.ssh/config` 别名：10.123.1.95:6004、user lc、IdentityFile `~/.ssh/lc_fluoresfm`）。云端工作目录 `/mnt/ssd3/lc/FluoResFM`（即 `~/FluoResFM`，不是 `~/lc/FluoResFM`）。
- 环境：云端 `.venv`（Python 3.10.12、torch 2.6.0+cu124、numpy 2.2.6、skimage 0.25.2）。**推理必须加离线环境变量** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`（HF Hub 不可达，BiomedCLIP tokenizer 已缓存于 `~/.cache/huggingface`；不带会因 `get_tokenizer` 联网超时失败）。
- 同步（探针最小集 / 完整 example，均需 `--force-scp`）：
  ```bash
  python scripts/sync_probe_to_server.py --host lc-Dual-4090 --user lc --port 6004 --remote-dir "~/FluoResFM" --include-probe-data --force-scp
  python scripts/sync_probe_to_server.py --host lc-Dual-4090 --user lc --port 6004 --remote-dir "~/FluoResFM" --include-data --force-scp
  ```
- 运行（长任务 SSH 前台会超时被杀，用 `nohup ... &` + 轮询 `manifest.json`/`failure.json`）：
  ```bash
  cd /mnt/ssd3/lc/FluoResFM
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup .venv/bin/python scripts/probe_biosr_mt_fluoresfm_server.py \
    --config <cfg> --workspace-root ~/FluoResFM > /tmp/run.log 2>&1 &
  ```
- GPU 纪律：双 RTX 4090；跑前 `nvidia-smi` 确认无其他用户进程（他人占用则暂停）；**批量生产固定在单一 GPU 上执行**（跨 GPU 有设备级低阶位差）。
- 关键哈希（云端=本地，阶段 3 已核对）：checkpoint `epoch_0_iter_700000.pt`=`d2402bff6e52fce05e5442e96eb0a9a39623adc932c73b46ddd8998d2f452db3`；embedder `open_clip_config.json`=`9a41f334a8c444678772c0ebb9ab854c97ab350bced3a17b803e258d39c23dc0`、`open_clip_pytorch_model.bin`=`52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be`；输入 `WF_noise_level_3/41.tif`=`8c437b43b68a3281d060adc821b57566ddbaedf3f12d48fc0b67b2e4120dc202`；参考 `WF_noise_level_3_fluoresfm/41.tif`=`db01b963bd3d4de55bf43562cb79677ebcb555b283978ba11e97bc360c3f0f4b`。
- 版本锁定：napari-fluoresfm v0.3.4（`2fded7c31be8c52476de89934ced9093ebe2c307`）；生产配置 `configs/biosr_mt_fluoresfm_production_v1.json`（patch=64、batch=8、compile=false、sf_lr=1、无 flash）。

## 最终验收

1. 参考基线 83,591 个文件全部在最终 manifest 中有唯一状态：严格生产、数值等价、未通过或明确排除；不得遗漏或重复计数。
2. 550 个全图、82,200 个训练 patch、16 个文本文件保留阶段 2 证据；90 个 `_fluoresfm` 全图与 735 个测试 patch 以阶段 3–4 新证据补齐。
3. 执行配置 JSON 解析、路径策略检查、脚本语法检查、逐文件比较、Markdown 本地链接检查与 Git whitespace 检查。
4. 形成新的中文审计报告，清楚区分：严格文件一致、浮点数值等价、失败/待定项及其限制。

## 已知风险与停止条件

- 历史生成时的 Torch、Triton、CUDA、模型源码、prompt 或 compile 行为任一项不同，都可能使推理 TIFF 不一致；不能由近似视觉结果推断严格成功。
- 缺失的历史运行清单、环境锁或精确 prompt 可能导致多个候选链均不通过。此时应报告“无法在现有证据下严格生产”，而不是选择误差最小的候选。
- 不在服务器上安装、升级或替换共享系统依赖；需要兼容环境时，在个人工作区或经管理员批准的容器/Conda 环境中建立隔离环境。
- 任一阶段失败保留独立候选目录与日志，禁止删除参考、覆盖旧证据或以失败候选替代 bundled example。
