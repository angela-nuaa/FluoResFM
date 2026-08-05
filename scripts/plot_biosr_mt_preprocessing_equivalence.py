#!/usr/bin/env python
"""Create visual evidence panels for the BioSR_MT preprocessing audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "experiment_results" / "figures"
READER = ROOT / "data" / "raw" / "BioTISR" / "Supplementary Files for BioTISR" / "IO_MRC_Python" / "read_mrc.py"


def read_mrc(path: Path) -> np.ndarray:
    spec = importlib.util.spec_from_file_location("biosr_mt_plot_reader", READER)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法导入 MRC 读取器。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, array = module.read_mrc(str(path))
    return array.astype(np.float32)


def limits(image: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(image, (1, 99.8)))  # type: ignore[return-value]


def axis_image(axis: plt.Axes, image: np.ndarray, title: str, *, cmap: str = "magma", clim: tuple[float, float] | None = None) -> None:
    kwargs = {} if clim is None else {"vmin": clim[0], "vmax": clim[1]}
    axis.imshow(image, cmap=cmap, **kwargs)
    axis.set_title(title, fontsize=10)
    axis.set_xticks([]); axis.set_yticks([])


def normalized(image: np.ndarray) -> np.ndarray:
    image = np.clip(image.astype(np.float32), 0, None)
    low, high = np.percentile(image, (3, 99.5))
    return (image - low) / (high - low)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test_raw = ROOT / "data" / "raw" / "BioSR" / "BioSR_MT" / "cells_041_055_all_levels" / "Microtubules" / "Cell_041" / "RawSIMData_level_03.mrc"
    test_example = ROOT / "example" / "data" / "BioSR_MT" / "test" / "channel_0" / "WF_noise_level_3" / "41.tif"
    raw = read_mrc(test_raw)
    mean = raw.mean(axis=2, dtype=np.float32)
    example = tifffile.imread(test_example).squeeze().astype(np.float32)
    clim = limits(mean)
    figure, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for frame, axis in enumerate(axes.flat[:9]):
        axis_image(axis, raw[:, :, frame], f"原始帧 {frame + 1}/9", clim=clim)
    axis_image(axes.flat[9], mean, "9 帧均值（复建）", clim=clim)
    axis_image(axes.flat[10], example, "example WF_noise_level_3", clim=clim)
    difference = np.abs(mean - example)
    axis_image(axes.flat[11], difference, "绝对差（最大值 = 0）", cmap="viridis", clim=(0, 1))
    figure.suptitle("BioSR_MT 测试样本 41：RawSIM level 03 → example 输入", fontsize=14, fontweight="bold")
    figure.savefig(OUT / "预处理实验-01_biosr_mt_test_raw_to_example_41_level03.png", dpi=180)
    plt.close(figure)

    train_dir = ROOT / "data" / "raw" / "BioSR" / "BioSR_MT" / "cells_001_040_all_levels" / "Microtubules" / "Cell_001"
    lr_raw = read_mrc(train_dir / "RawSIMData_level_03.mrc")
    hr_raw = read_mrc(train_dir / "SIM_gt.mrc")[:, :, 0]
    lr_full = lr_raw.mean(axis=2, dtype=np.float32)
    lr_reference = tifffile.imread(ROOT / "example" / "data" / "BioSR_MT" / "train" / "channel_0" / "WF_noise_level_3" / "1.tif").squeeze().astype(np.float32)
    lr_norm, hr_norm = normalized(lr_full), normalized(hr_raw)
    row, col = 7, 7
    lr_patch = lr_norm[row * 32 : row * 32 + 32, col * 32 : col * 32 + 32]
    hr_patch = hr_norm[row * 64 : row * 64 + 64, col * 64 : col * 64 + 64]
    lr_patch_ref = tifffile.imread(ROOT / "example" / "data" / "BioSR_MT" / "train" / "channel_0" / "WF_noise_level_3_p32_s32_2d" / f"1_{row}_{col}.tif").squeeze().astype(np.float32)
    hr_patch_ref = tifffile.imread(ROOT / "example" / "data" / "BioSR_MT" / "train" / "channel_0" / "SIM_p64_s64_2d" / f"1_{row}_{col}.tif").squeeze().astype(np.float32)
    figure, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    full_clim = limits(lr_full)
    axis_image(axes[0, 0], lr_full, "RawSIM 9 帧均值", clim=full_clim)
    axis_image(axes[0, 1], lr_reference, "example 全图", clim=full_clim)
    axis_image(axes[0, 2], np.abs(lr_full - lr_reference), "全图绝对差（最大值 = 0）", cmap="viridis", clim=(0, 1))
    axis_image(axes[0, 3], hr_raw, "SIM_gt 第 0 平面", clim=limits(hr_raw))
    axis_image(axes[1, 0], lr_patch, "复建并归一化的 LR patch 32×32", clim=(-0.1, 1.1))
    axis_image(axes[1, 1], lr_patch_ref, "example LR patch 32×32", clim=(-0.1, 1.1))
    axis_image(axes[1, 2], np.abs(lr_patch - lr_patch_ref), "LR patch 绝对差", cmap="viridis", clim=(0, 1e-6))
    axis_image(axes[1, 3], np.abs(hr_patch - hr_patch_ref), "HR patch 绝对差", cmap="viridis", clim=(0, 1e-6))
    figure.suptitle("BioSR_MT 训练样本 1、level 03：全图与 patch 复建对照（patch 7,7）", fontsize=14, fontweight="bold")
    figure.savefig(OUT / "预处理实验-01_biosr_mt_train_full_and_patch_01_level03.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
