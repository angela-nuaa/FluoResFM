"""Compute stable reference-based metrics for prompt-ablation TIFF outputs.

This intentionally omits NanoPyx SQUIRREL and decorrelation analysis.  Those
optional metrics are evaluated separately because they can be resource-heavy;
their failure must not be mistaken for an inference failure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
import torch
from pytorch_msssim import ms_ssim
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, structural_similarity


def normalize(image: np.ndarray, low: float = 0.03, high: float = 0.995) -> np.ndarray:
    lo, hi = np.quantile(image.astype(np.float32), (low, high))
    if not hi > lo:
        raise ValueError("Image has insufficient intensity range for normalisation.")
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def zncc(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = reference.ravel() - reference.mean()
    pred = prediction.ravel() - prediction.mean()
    denominator = np.linalg.norm(ref) * np.linalg.norm(pred)
    return float(np.dot(ref, pred) / denominator) if denominator else float("nan")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--dataset", choices=["BioSR_MT", "BioTISR_CCP"], default="BioSR_MT")
    parser.add_argument("--runs-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, runs_dir = args.repo_root.resolve(), args.runs_dir.resolve()
    reference_dir = {
        "BioSR_MT": root / "example" / "data" / "BioSR_MT" / "test" / "channel_0" / "SIM",
        "BioTISR_CCP": root / "example" / "data" / "BioTISR_CCP" / "train" / "channel_0" / "SIM",
    }[args.dataset]
    conditions = sorted(path for path in runs_dir.glob("[0-9][0-9]_*") if path.is_dir())
    if not conditions:
        raise FileNotFoundError(f"No condition directories found in {runs_dir}")

    rows: list[dict[str, object]] = []
    for condition in conditions:
        for prediction_path in sorted(condition.glob("*.tif")):
            reference_path = reference_dir / prediction_path.name
            reference = normalize(tifffile.imread(reference_path)).squeeze()
            prediction = normalize(tifffile.imread(prediction_path)).squeeze()
            if reference.shape != prediction.shape:
                raise ValueError(f"Shape mismatch for {prediction_path.name}: {prediction.shape} vs {reference.shape}")
            rows.append(
                {
                    "condition": condition.name,
                    "image": prediction_path.name,
                    "psnr": peak_signal_noise_ratio(reference, prediction, data_range=1.0),
                    "ssim": structural_similarity(reference, prediction, data_range=1.0),
                    "msssim": float(ms_ssim(torch.from_numpy(reference)[None, None], torch.from_numpy(prediction)[None, None], data_range=1.0).item()),
                    "zncc": zncc(reference, prediction),
                    "nrmse": normalized_root_mse(reference, prediction),
                }
            )

    fields = ["condition", "image", "psnr", "ssim", "msssim", "zncc", "nrmse"]
    metrics_path = runs_dir / "basic_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = runs_dir / "basic_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["condition", "n", "psnr_mean", "ssim_mean", "msssim_mean", "zncc_mean", "nrmse_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in sorted({str(row["condition"]) for row in rows}):
            subset = [row for row in rows if row["condition"] == condition]
            writer.writerow(
                {
                    "condition": condition,
                    "n": len(subset),
                    "psnr_mean": np.mean([float(row["psnr"]) for row in subset]),
                    "ssim_mean": np.mean([float(row["ssim"]) for row in subset]),
                    "msssim_mean": np.mean([float(row["msssim"]) for row in subset]),
                    "zncc_mean": np.mean([float(row["zncc"]) for row in subset]),
                    "nrmse_mean": np.mean([float(row["nrmse"]) for row in subset]),
                }
            )
    print(f"Wrote {metrics_path} and {summary_path}")


if __name__ == "__main__":
    main()
