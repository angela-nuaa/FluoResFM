# 协议：BioSR-MT FluoResFM 推理生产与验收

> 依据：推理实验-01..04（结果记录）；本协议是把它们确定下来的规范收敛为可复用标准。
> 范围：BioSR-MT bundled example 的 `_fluoresfm` 推理输出生产与验收；判定标准为**质量等价**（跨进程/跨 GPU 无法逐字节复现他人字节）。
> 指标口径：官方口径——每图 P3/P99.5 归一化 → clip[0,2.5] → PSNR/SSIM `data_range=2.5`（论文 `4_0_result_evaluate.py`）。

## 1. 评估口径（官方协议）

- 每图独立 P3/P99.5 归一化 → `np.clip(x, 0, 2.5)` → skimage `PSNR`/`SSIM` `data_range=2.5`。
- vs-SIM：SIM 用官方 `interp_sf(sf=-2)`（`avg_pool2d(kernel=2)`）降到 502 对齐输出分辨率。
- **无背景扣除（bkg_sub）**：官方代码的 rolling-ball 背景扣除在 bundled 502×502 上崩溃（496≠502），且论文方法节无此描述；bundled 尺度保持无 bkg 自洽口径。
- 官方 `num_sample=8` vs 15 图均值差 <0.2 dB，可直接比对。

## 2. 生产配置（冻结参数）

| 项 | 值 |
| --- | --- |
| 版本 | napari-fluoresfm **v0.3.4**（`2fded7c31be8c52476de89934ced9093ebe2c307`）|
| patch_size / overlap | **64 / 16**（patch//4 自动）|
| batch_size | 8 |
| compile | false |
| sf_lr | 1（去卷积）|
| flash_attention | false（p64 注意力矩阵小，无需 flash）|
| prompt | 完整 8 字段（≡ 官方 `biosr-mt-dcv-3` 元数据）：`Task: deconvolution; sample: fixed COS-7 cell line; structure: microtubule; fluorescence indicator: mEmerald (GFP); input microscope: wide-field microscope with excitation numerical aperture (NA) of 1.35, detection numerical aperture (NA) of 1.3; input pixel size: 62.6 x 62.6 nm; target microscope: linear structured illumination microscope with excitation numerical aperture (NA) of 1.35, detection numerical aperture (NA) of 1.3; target pixel size: 62.6 x 62.6 nm.` |
| 环境变量 | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`（HF Hub 不可达，tokenizer 已缓存）|
| 设备纪律 | 跑前 `nvidia-smi` 确认无他人进程；批量固定在单一 GPU |

配置入口：`configs/biosr_mt_fluoresfm_production_v1.json`（生产）、`configs/biosr_mt_fluoresfm_batch_production_v1.json`（批量）、`_deterministic_v1.json`（确定性）。

## 3. 验收标准（质量等价）

- **全图**：候选 vs bundled 参考 path/shape/dtype 一致 + 官方口径 **PSNR ≥45 dB 且 SSIM ≥0.99**（带裕度；观测下限 46.02/0.9965，跨运行有 ~0.3 dB / ~0.0003 波动）。
- **测试 patch**（派生资产）：结构匹配（shape/dtype/735 数量/命名）+ 逐 patch 官方口径 **均值 ≥44 dB**（观测 46.02，min 26.8 / median 46.5）；逐 patch 分布如实报告，不标文件级严格一致。
- **逐文件验证**：`comparison.csv` + `manifest.json`（含环境冻结、源码/权重哈希、参数、判定）；拒绝覆盖。
- **检查套件**：JSON 解析、脚本语法、配置路径策略、Markdown 链接、Git whitespace。

## 4. 确定性约定

- **同进程内**：逐字节确定。
- **跨进程（同 GPU 同配置）**：不确定（~2⁻¹⁰，质量 80+ dB）。根因：TF32 CuBLAS 非确定性 reduction + `cudnn.benchmark=True` 跨进程算法选择。
- **跨 GPU**：设备级低阶位差（~2⁻⁹，77 dB+）。
- **可选确定性变体**（跨进程逐字节可复现，磁盘源码不变）：`CUBLAS_WORKSPACE_CONFIG=:4096:8` + `torch.use_deterministic_algorithms(True)` + 运行时注入 benchmark=False 的 predict（`produce_biosr_mt_fluoresfm_batch.py` 内置，config 设 `deterministic: true`）。

## 5. 参考基线与证据入口

| 项 | 值 |
| --- | --- |
| level_3 15 图基线 | 候选 vs 参考 **48.13 dB / 0.9974**（最低 45.99）|
| vs-SIM（官方 avg-pool 口径）| 15 图 level_3：候选 34.76 / 参考 34.58 / 输入 27.56（+7.20 dB）；90 图全级别：候选 34.68 ≈ 参考 34.63 |
| 全目录验收 | **83,591 文件**：566 严格 + 82,200 数值等价 + 825 质量等价，无遗漏/重复/排除 |
| 批量输出 | 云端 `20260808_batch_production_v1/`（主）、`_det_v1/`（确定性）|
| 汇总 | `data/derived/biosr-mt-final-acceptance/20260808/final/`（`final_manifest.json` + `final_status.csv`）|

入口脚本：`scripts/produce_biosr_mt_fluoresfm_batch.py`（生产）、`scripts/final_acceptance_biosr_mt.py`（验收汇总）、`scripts/eval_biosr_mt_official_metrics.py`（口径复核）、`scripts/eval_biosr_mt_batch_vs_sim.py`（SIM 锚定）、`scripts/diag_numeric_variation.py`（确定性诊断）。

## 6. 对新数据集的使用流程

1. 确认输入与参考 TIFF（shape/dtype/哈希），按 §1 口径准备。
2. 按 §2 冻结配置运行推理（`produce_biosr_mt_fluoresfm_batch.py`）。
3. 按 §3 验收（全图 PSNR/SSIM 带裕度阈值；派生 patch 资产级判定），产出 manifest + comparison.csv。
4. 如需跨进程逐字节复现，用 §4 确定性变体。
5. 更新 §5 基线，并写结果记录（参考推理实验-01..04 结构）。
