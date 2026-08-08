#!/usr/bin/env python
"""Generate auditable figure evidence for inference experiments 01--04.

The plots intentionally use only checked-in tables/manifests (and the values
recorded in the corresponding experiment reports).  They therefore remain
reproducible without the unavailable cloud GPU output directories.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "experiment_results" / "figures"
ACCEPTANCE = ROOT / "data" / "derived" / "biosr-mt-final-acceptance" / "20260808"
EXAMPLE = ROOT / "example" / "data" / "BioSR_MT" / "test" / "channel_0"
VISUAL_CANDIDATES = ACCEPTANCE / "visual_evidence" / "p64_deterministic_level3"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 160,
})


def save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def experiment_01() -> None:
    """Parameter-selection evidence transcribed from experiment 01/02."""
    patch = [64, 256]
    psnr = [48.13, 42.00]
    ssim = [0.9974, 0.9915]
    colors = ["#2374ab", "#a7c6da"]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)
    bars = axes[0].bar([str(item) for item in patch], psnr, color=colors, width=0.62)
    axes[0].axhline(46.0, color="#bc4749", linestyle="--", linewidth=1.2, label="quality floor (46 dB)")
    axes[0].set(title="Candidate vs bundled reference", xlabel="Patch size", ylabel="PSNR (dB)", ylim=(38, 51))
    for bar, value in zip(bars, psnr):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.32, f"{value:.2f}", ha="center", fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=8)
    bars = axes[1].bar([str(item) for item in patch], ssim, color=colors, width=0.62)
    axes[1].set(title="Structural similarity", xlabel="Patch size", ylabel="SSIM", ylim=(0.985, 1.000))
    for bar, value in zip(bars, ssim):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.00025, f"{value:.4f}", ha="center", fontweight="bold")
    figure.suptitle("Inference experiment 01 — 15-image parameter-selection evidence", fontweight="bold")
    figure.text(0.5, -0.02, "Official per-image normalization/clip protocol; values reported in experiments 01–02.", ha="center", fontsize=8)
    save(figure, "推理实验-01_patch设置质量证据.png")


def experiment_02() -> None:
    """Plot the complete 15-image p64 reproduction result from experiment 02."""
    image_ids = np.arange(41, 56)
    psnr = np.array([47.27, 49.14, 49.02, 49.88, 48.02, 47.55, 46.64, 46.88, 48.60, 48.80, 45.99, 48.55, 48.18, 48.89, 48.57])
    ssim = np.array([0.9969, 0.9975, 0.9976, 0.9982, 0.9980, 0.9968, 0.9971, 0.9972, 0.9974, 0.9975, 0.9974, 0.9975, 0.9965, 0.9975, 0.9972])
    figure, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, constrained_layout=True)
    axes[0].plot(image_ids, psnr, marker="o", color="#2374ab", linewidth=1.8)
    axes[0].axhline(psnr.mean(), color="#2374ab", linestyle="--", label=f"mean = {psnr.mean():.2f} dB")
    axes[0].axhline(46.0, color="#bc4749", linestyle="--", label="quality floor = 46 dB")
    axes[0].set(title="p64 candidate vs bundled reference", ylabel="PSNR (dB)", ylim=(45, 51))
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].plot(image_ids, ssim, marker="o", color="#3a9d5d", linewidth=1.8)
    axes[1].axhline(ssim.mean(), color="#3a9d5d", linestyle="--", label=f"mean = {ssim.mean():.4f}")
    axes[1].set(xlabel="Test image ID", ylabel="SSIM", ylim=(0.996, 0.999))
    axes[1].set_xticks(image_ids)
    axes[1].legend(loc="lower right", fontsize=8)
    figure.suptitle("Inference experiment 02 — complete 15-image reproduction evidence", fontweight="bold")
    save(figure, "推理实验-02_15图复现逐图指标.png")


def read_full_image_psnr() -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    with (ACCEPTANCE / "comparison_main.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["asset"] == "full_image":
                values[row["level"]].append(float(row["psnr_db"]))
    if sum(map(len, values.values())) != 90:
        raise ValueError("comparison_main.csv must contain 90 full-image rows")
    return dict(sorted(values.items(), key=lambda item: int(item[0].rsplit("_", 1)[1])))


def experiment_03() -> None:
    """Show final all-level reproduction and SIM anchor evidence."""
    full_image = read_full_image_psnr()
    levels = list(full_image)
    labels = [item.rsplit("_", 1)[1] for item in levels]
    candidate = [32.96, 34.17, 34.76, 34.96, 35.11, 36.10]
    reference = [32.95, 34.16, 34.58, 34.92, 35.08, 36.06]
    input_wf = [26.44, 27.29, 27.56, 27.60, 27.50, 27.62]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].boxplot([full_image[level] for level in levels], tick_labels=[f"L{label}" for label in labels], showmeans=True)
    axes[0].axhline(46.0, color="#bc4749", linestyle="--", linewidth=1.2, label="quality floor")
    axes[0].set(title="90 final full images vs bundled reference", xlabel="Noise level", ylabel="PSNR (dB)", ylim=(44, 62))
    axes[0].legend(loc="lower right", fontsize=8)
    x = np.arange(len(levels)); width = 0.25
    axes[1].bar(x - width, input_wf, width, label="WF input", color="#a7c6da")
    axes[1].bar(x, reference, width, label="Bundled reference", color="#8db580")
    axes[1].bar(x + width, candidate, width, label="Production candidate", color="#2374ab")
    axes[1].set(title="Independent SIM anchor (90 images)", xlabel="Noise level", ylabel="PSNR vs SIM (dB)", xticks=x, xticklabels=[f"L{label}" for label in labels], ylim=(24, 38))
    axes[1].legend(fontsize=8, loc="upper left")
    figure.suptitle("Inference experiment 03 — all-level and truth-anchored evidence", fontweight="bold")
    figure.text(0.5, -0.02, "Left: data/derived/.../comparison_main.csv. Right: reported official avg-pool SIM evaluation.", ha="center", fontsize=8)
    save(figure, "推理实验-03_全级别复现与SIM锚定.png")


def experiment_04() -> None:
    """Visualize the final one-state-per-reference-file acceptance manifest."""
    with (ACCEPTANCE / "final" / "final_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    counts = manifest["counts"]
    status = counts["status"]
    family = counts["family"]
    status_labels = ["Byte-identical", "Float equivalent", "Quality equivalent"]
    status_values = [status["严格文件一致"], status["浮点数值等价"], status["质量等价"]]
    colors = ["#5ca4a9", "#9bc53d", "#f6c85f"]
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), constrained_layout=True)
    status_bars = axes[0].barh(status_labels, status_values, color=colors)
    axes[0].set(title=f"Final status ({counts['total_files']:,} files)", xlabel="Files", xscale="log")
    for bar, value in zip(status_bars, status_values):
        axes[0].text(value * 1.18, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontsize=9)
    names = ["WF/SIM full", "Text", "Inference full", "Test patches", "Train patches"]
    values = [family["550 全图(WF/SIM)"], family["16 文本"], family["90 *_fluoresfm 全图"], family["735 测试 patch"], family["82,200 训练 patch"]]
    bars = axes[1].barh(names, values, color=["#5ca4a9", "#5ca4a9", "#f6c85f", "#f6c85f", "#9bc53d"])
    axes[1].set(title="Reference inventory covered by asset family", xlabel="Files", xscale="log")
    for bar, value in zip(bars, values):
        axes[1].text(value * 1.15, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontsize=9)
    figure.suptitle("Inference experiment 04 — final acceptance-manifest evidence", fontweight="bold")
    figure.text(0.5, -0.02, "Manifest verifies no omission, no duplication, and no exclusion; verdict = pass.", ha="center", fontsize=8)
    save(figure, "推理实验-04_最终验收状态与资产覆盖.png")


def roi_from_structure(image: np.ndarray, size: int = 128) -> tuple[int, int]:
    """Choose a reproducible high-gradient ROI using only the reference image."""
    image = image.squeeze().astype(np.float32)
    gradient = np.zeros_like(image)
    gradient[:, 1:] += np.abs(np.diff(image, axis=1))
    gradient[1:, :] += np.abs(np.diff(image, axis=0))
    integral = np.pad(gradient, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    best_score, best = -np.inf, (0, 0)
    for row in range(0, image.shape[0] - size + 1, 8):
        for column in range(0, image.shape[1] - size + 1, 8):
            score = integral[row + size, column + size] - integral[row, column + size] - integral[row + size, column] + integral[row, column]
            if score > best_score:
                best_score, best = score, (row, column)
    return best


def official_display_normalize(image: np.ndarray) -> np.ndarray:
    """Use the report's per-image P3/P99.5 display normalization."""
    image = image.astype(np.float32)
    low, high = np.percentile(image, (3, 99.5))
    return np.clip((image - low) / (high - low), 0, 2.5)


def experiment_01_real_images() -> None:
    """Render authentic final p64 candidate images from the synchronized run."""
    image_ids = [41, 44, 47]
    if not all((VISUAL_CANDIDATES / f"{image_id}.tif").is_file() for image_id in image_ids):
        raise FileNotFoundError("Missing synchronized p64 TIFFs; see data/derived/.../visual_evidence.")
    metrics: dict[str, dict[str, str]] = {}
    with (ACCEPTANCE / "comparison_det.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["asset"] == "full_image" and row["level"] == "WF_noise_level_3" and int(row["image_id"]) in image_ids:
                metrics[row["image_id"]] = row
    figure, axes = plt.subplots(len(image_ids), 6, figsize=(15, 8.2), constrained_layout=True)
    headings = ["WF input", "p64 candidate", "Bundled reference", "|candidate − reference|", "Candidate ROI", "Reference ROI"]
    for row_index, image_id in enumerate(image_ids):
        wf = tifffile.imread(EXAMPLE / "WF_noise_level_3" / f"{image_id}.tif").squeeze().astype(np.float32)
        candidate = tifffile.imread(VISUAL_CANDIDATES / f"{image_id}.tif").squeeze().astype(np.float32)
        reference = tifffile.imread(EXAMPLE / "WF_noise_level_3_fluoresfm" / f"{image_id}.tif").squeeze().astype(np.float32)
        difference = np.abs(candidate - reference)
        difference_high = np.percentile(difference, 99.9)
        roi_row, roi_column = roi_from_structure(reference)
        roi = np.s_[roi_row : roi_row + 128, roi_column : roi_column + 128]
        wf_display = official_display_normalize(wf)
        candidate_display = official_display_normalize(candidate)
        reference_display = official_display_normalize(reference)
        panels = [wf_display, candidate_display, reference_display, difference, candidate_display[roi], reference_display[roi]]
        for column, (axis, panel) in enumerate(zip(axes[row_index], panels)):
            if column == 3:
                axis.imshow(panel, cmap="viridis", vmin=0, vmax=difference_high)
            else:
                axis.imshow(panel, cmap="magma", vmin=0, vmax=1.2)
            axis.set_xticks([]); axis.set_yticks([])
            if row_index == 0:
                axis.set_title(headings[column], fontsize=10)
            if column in (0, 1, 2):
                rectangle = plt.Rectangle((roi_column, roi_row), 128, 128, fill=False, edgecolor="cyan", linewidth=0.8)
                axis.add_patch(rectangle)
        metric = metrics[str(image_id)]
        axes[row_index, 0].set_ylabel(
            f"L3 / {image_id}.tif\\nPSNR {float(metric['psnr_db']):.2f} dB\\nSSIM {float(metric['ssim']):.4f}",
            fontsize=9,
        )
    figure.suptitle("Inference experiment 01 — authentic final p64 image comparison", fontweight="bold")
    figure.text(0.5, -0.02, "WF/candidate/reference each use the report's P3/P99.5 normalization (shared display range 0–1.2); residual is independently amplified to P99.9.", ha="center", fontsize=8)
    save(figure, "推理实验-02_真实图像对照_p64_level3.png")


def main() -> None:
    experiment_01()
    experiment_01_real_images()
    experiment_02()
    experiment_03()
    experiment_04()


if __name__ == "__main__":
    main()
