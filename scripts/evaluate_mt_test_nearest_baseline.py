#!/usr/bin/env python
"""Evaluate the 2× nearest-neighbour baseline on bundled BioSR_MT test images."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
import torch
from nanopyx.core.analysis.decorr import DecorrAnalysis
from nanopyx.core.transform import ErrorMap
from pytorch_msssim import ms_ssim
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, structural_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--pixel-size-nm", type=float, default=31.3)
    parser.add_argument("--prediction-dir", type=Path, help="可选：评估已保存的预测目录；省略时计算 2× 最近邻基线。")
    parser.add_argument("--condition", default="nearest_2x", help="写入结果的条件名称。")
    return parser.parse_args()


def normalise(image: np.ndarray) -> np.ndarray:
    low, high = np.quantile(image.astype(np.float32), (0.03, 0.995))
    if not high > low:
        raise ValueError("图像强度范围不足，无法进行百分位归一化。")
    return np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def zncc(reference: np.ndarray, prediction: np.ndarray) -> float:
    left, right = reference.ravel() - reference.mean(), prediction.ravel() - prediction.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else float("nan")


def calculate(path_input: str, path_reference: str, path_prediction: str | None, condition: str, pixel_size_nm: float) -> dict[str, object]:
    filename = Path(path_input).name
    lr = tifffile.imread(path_input).squeeze().astype(np.float32)
    reference = tifffile.imread(path_reference).squeeze().astype(np.float32)
    prediction = tifffile.imread(path_prediction).squeeze().astype(np.float32) if path_prediction is not None else np.repeat(np.repeat(lr, 2, axis=0), 2, axis=1)
    if prediction.shape != reference.shape:
        raise ValueError(f"{filename} 的最近邻预测与参考图尺寸不一致：{prediction.shape} 对 {reference.shape}")
    reference, prediction = normalise(reference), normalise(prediction)
    error_map = ErrorMap()
    error_map.optimise(reference.astype(np.float32), prediction.astype(np.float32))
    da = DecorrAnalysis(pixel_size=pixel_size_nm, units="nm")
    da.run_analysis(prediction.astype(np.float32))
    return {
        "condition": condition,
        "image": filename,
        "psnr": float(peak_signal_noise_ratio(reference, prediction, data_range=1.0)),
        "ssim": float(structural_similarity(reference, prediction, data_range=1.0)),
        "msssim": float(ms_ssim(torch.from_numpy(reference)[None, None], torch.from_numpy(prediction)[None, None], data_range=1.0).item()),
        "zncc": zncc(reference, prediction),
        "nrmse": float(normalized_root_mse(reference, prediction)),
        "rse": float(error_map.getRSE()),
        "rsp": float(error_map.getRSP()),
        "resolution_da_nm": float(da.resolution),
    }


def initialise_worker() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖既有目录：{args.output_dir}")
    if args.workers < 1:
        raise ValueError("--workers 必须为正整数。")
    root = args.repo_root.resolve()
    input_dir = root / "example" / "data" / "BioSR_MT" / "test" / "channel_0" / "WF_noise_level_3"
    reference_dir = root / "example" / "data" / "BioSR_MT" / "test" / "channel_0" / "SIM"
    filenames = [line.strip() for line in (root / "example" / "data" / "BioSR_MT" / "test.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    prediction_dir = args.prediction_dir.resolve() if args.prediction_dir is not None else None
    if prediction_dir is not None and any(not (prediction_dir / filename).is_file() for filename in filenames):
        raise FileNotFoundError("预测目录必须包含 test.txt 中列出的全部 TIFF 文件。")
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=initialise_worker) as executor:
        futures = [executor.submit(calculate, str(input_dir / filename), str(reference_dir / filename), str(prediction_dir / filename) if prediction_dir is not None else None, args.condition, args.pixel_size_nm) for filename in filenames]
        for complete, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"已完成图像：{complete}/{len(filenames)}", flush=True)
    rows.sort(key=lambda row: str(row["image"]))
    fields = list(rows[0])
    with (args.output_dir / "metrics_nearest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    summary = {f"mean_{field}": float(np.mean([float(row[field]) for row in rows])) for field in fields[2:]}
    summary["n_images"] = len(rows)
    (args.output_dir / "summary_nearest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source = "2× 最近邻基线" if prediction_dir is None else f"已保存预测：{prediction_dir}"
    run = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": f"BioSR_MT 示例测试集 15 张图的 {source} 指标补算。", "normalisation": "参考图与预测分别按 P3/P99.5 归一化至 [0,1]。", "pixel_size_nm": args.pixel_size_nm, "workers": args.workers, "condition": args.condition, "n_images": len(rows)}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
