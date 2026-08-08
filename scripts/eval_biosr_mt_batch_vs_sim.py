#!/usr/bin/env python
"""SIM 锚定的批量生产候选 vs 官方推理等价性验证（CPU-only）。

对 6 个噪声级别共 90 张图，分别计算：
  - 生产候选 vs SIM（官方口径）
  - bundled 参考（官方推理输出）vs SIM
  - WF 输入 vs SIM（基线）

若候选 vs SIM ≈ 参考 vs SIM，则生产输出的真值保真度与官方推理等价（独立于参考本身）。

口径：SIM 用官方 interp_sf(sf=-2)=avg_pool2d(k=2) 降到 502 对齐；每图 P3/P99.5 →
clip[0,2.5] → PSNR/SSIM data_range=2.5（官方 4_0_result_evaluate.py，无背景扣除）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import skimage.io as io
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

LEVELS = ["WF_noise_level_1", "WF_noise_level_2", "WF_noise_level_3", "WF_noise_level_4", "WF_noise_level_5", "WF_noise_level_7"]
IDS = list(range(41, 56))


def normalize(img: np.ndarray) -> np.ndarray:
    vmin = np.percentile(img, 3)
    vmax = np.percentile(img, 99.5)
    if vmax == 0:
        vmax = img.max()
    amp = vmax - vmin
    if amp == 0:
        amp = 1
    return (img - vmin) / amp


def prep(img: np.ndarray) -> np.ndarray:
    out = np.clip(normalize(img), 0.0, 2.5)
    return out[0] if out.ndim == 3 and out.shape[0] == 1 else out


def avg_pool2d(x: np.ndarray, k: int) -> np.ndarray:
    return torch.nn.functional.avg_pool2d(torch.tensor(x), kernel_size=k, stride=k).numpy()


def ps(a: np.ndarray, b: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(a, b, data_range=2.5))


def ss(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, data_range=2.5))


def main(workspace_root: Path) -> None:
    ch = workspace_root / "example/data/BioSR_MT/test/channel_0"
    cand_root = workspace_root / "experiments/preprocess-03_biosr-mt-fluoresfm-inference-audit/20260808_batch_production_v1/full_images"
    print(f"{'level':18s} {'cand_vs_sim':>12s} {'ref_vs_sim':>12s} {'delta':>7s} {'inp_vs_sim':>12s}")
    overall_c, overall_r, overall_i = [], [], []
    for level in LEVELS:
        cand_ps, ref_ps, inp_ps = [], [], []
        for i in IDS:
            name = f"{i}.tif"
            cand = io.imread(cand_root / level / name).astype(np.float32)
            ref = io.imread(ch / f"{level}_fluoresfm" / name).astype(np.float32)
            inp = io.imread(ch / level / name).astype(np.float32)
            sim = io.imread(ch / "SIM" / name).astype(np.float32)
            sim_down = prep(avg_pool2d(sim, 2))
            cand_ps.append(ps(prep(cand), sim_down))
            ref_ps.append(ps(prep(ref), sim_down))
            inp_ps.append(ps(prep(inp), sim_down))
        cm, rm, im_ = float(np.mean(cand_ps)), float(np.mean(ref_ps)), float(np.mean(inp_ps))
        overall_c += cand_ps
        overall_r += ref_ps
        overall_i += inp_ps
        print(f"{level:18s} {cm:12.4f} {rm:12.4f} {cm - rm:+7.3f} {im_:12.4f}")
    oc, orr, oi = float(np.mean(overall_c)), float(np.mean(overall_r)), float(np.mean(overall_i))
    print("-" * 60)
    print(f"{'ALL 90':18s} {oc:12.4f} {orr:12.4f} {oc - orr:+7.3f} {oi:12.4f}")
    print(f"cand-vs-SIM over reference-vs-SIM: {oc - orr:+.3f} dB（<0.5 dB 视为等价）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace-root", default=Path.cwd(), type=Path, help="FluoResFM 仓库根目录")
    args = ap.parse_args()
    sys.exit(main(args.workspace_root))
