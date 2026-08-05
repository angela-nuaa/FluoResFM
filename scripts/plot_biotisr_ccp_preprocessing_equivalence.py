#!/usr/bin/env python
"""生成 BioTISR-CCP 原始 MRC 与 bundled example 的可视化对照。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "BioTISR" / "BioTISR_CCPs" / "Cell_001"
EXAMPLE = ROOT / "example" / "data" / "BioTISR_CCP" / "train" / "channel_0"
READER = ROOT / "data" / "raw" / "BioTISR" / "Supplementary Files for BioTISR" / "IO_MRC_Python" / "read_mrc.py"
OUT = ROOT / "docs" / "experiment_results" / "figures"


def load_reader():
    spec = importlib.util.spec_from_file_location("biotisr_plot_reader", READER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 MRC 读取器：{READER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_mrc


def as_tyx(stack: np.ndarray) -> np.ndarray:
    return np.moveaxis(stack, -1, 0).transpose(0, 2, 1).astype(np.float32)


def normalise(image: np.ndarray) -> np.ndarray:
    image = np.clip(image.astype(np.float32), 0, None)
    low, high = np.percentile(image, (3, 99.5))
    return (image - low) / (high - low if high > low else 1.0)


def limits(image: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(image, (1, 99.8)))  # type: ignore[return-value]


def image(axis: plt.Axes, array: np.ndarray, title: str, *, cmap: str = "magma", clim: tuple[float, float] | None = None) -> None:
    kwargs = {} if clim is None else {"vmin": clim[0], "vmax": clim[1]}
    axis.imshow(array, cmap=cmap, **kwargs)
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reader = load_reader()
    _, level_xyz = reader(str(RAW_ROOT / "RawSIMData_level_01.mrc"))
    level = as_tyx(level_xyz).reshape(20, 9, 512, 512)
    frames = level[0]
    wf = frames.mean(axis=0, dtype=np.float32)
    wf_example = tifffile.imread(EXAMPLE / "WF_noise_level_0_0" / "Cell_001_0.tif").squeeze().astype(np.float32)
    clim = limits(wf)

    figure, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for index, axis in enumerate(axes.flat[:9]):
        image(axis, frames[index], f"时间点 0：原始帧 {index + 1}/9", clim=clim)
    image(axes.flat[9], wf, "9 帧均值（复建）", clim=clim)
    image(axes.flat[10], wf_example, "example WF_noise_level_0", clim=clim)
    image(axes.flat[11], np.abs(wf - wf_example), "绝对差（最大值 = 0）", cmap="viridis", clim=(0, 1))
    figure.suptitle("BioTISR-CCP Cell_001：RawSIM level 01 → example WF", fontsize=14, fontweight="bold")
    figure.savefig(OUT / "预处理实验-02_biotisr_ccp_raw_frames_to_wf.png", dpi=180)
    plt.close(figure)

    _, gt_xyz = reader(str(RAW_ROOT / "SIM_gt.mrc"))
    sim = np.clip(as_tyx(gt_xyz)[0], 0, None)
    sim_example = tifffile.imread(EXAMPLE / "SIM_0" / "Cell_001_0.tif").squeeze().astype(np.float32)
    row = col = 7
    wf_patch = normalise(wf)[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32]
    sim_patch = normalise(sim)[row * 64 : (row + 1) * 64, col * 64 : (col + 1) * 64]
    wf_patch_example = tifffile.imread(EXAMPLE / "WF_noise_level_0_0_p32_s32_2d" / f"Cell_001_0_{row}_{col}.tif").squeeze().astype(np.float32)
    sim_patch_example = tifffile.imread(EXAMPLE / "SIM_0_p64_s64_2d" / f"Cell_001_0_{row}_{col}.tif").squeeze().astype(np.float32)

    figure, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    image(axes[0, 0], wf, "复建 WF 全图", clim=limits(wf))
    image(axes[0, 1], wf_example, "example WF 全图", clim=limits(wf))
    image(axes[0, 2], np.abs(wf - wf_example), "WF 全图差（最大值 = 0）", cmap="viridis", clim=(0, 1))
    image(axes[0, 3], sim, "SIM_gt 截断负值后", clim=limits(sim))
    image(axes[1, 0], sim_example, "example SIM 全图", clim=limits(sim))
    image(axes[1, 1], np.abs(sim - sim_example), "SIM 全图差（最大值 = 0）", cmap="viridis", clim=(0, 1))
    image(axes[1, 2], np.abs(wf_patch - wf_patch_example), "LR patch 差（≤2.43e-7）", cmap="viridis", clim=(0, 1e-6))
    image(axes[1, 3], np.abs(sim_patch - sim_patch_example), "HR patch 差（≤3.18e-7）", cmap="viridis", clim=(0, 1e-6))
    figure.suptitle("BioTISR-CCP Cell_001：全图与 SR 补丁复建对照（patch 7,7）", fontsize=14, fontweight="bold")
    figure.savefig(OUT / "预处理实验-02_biotisr_ccp_fullframe_and_patch.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
