"""Calculate screening metrics for prompt-ablation outputs against reference SIM.

Metrics are computed after independently percentile-normalising each prediction
and reference image to [0, 1]. They are useful for a controlled local
comparison, but are not a replacement for the paper's full evaluation protocol.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
from nanopyx.core.transform import ErrorMap
from nanopyx.core.analysis.decorr import DecorrAnalysis
from pytorch_msssim import ms_ssim
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, structural_similarity
import torch


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


def squirrel(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    """Return upstream NanoPyx SQUIRREL RSE and RSP on normalised images."""
    error_map = ErrorMap()
    error_map.optimise(reference.astype(np.float32), prediction.astype(np.float32))
    return float(error_map.getRSE()), float(error_map.getRSP())


def resolution_da(image: np.ndarray, pixel_size_nm: float) -> float:
    calculator = DecorrAnalysis(pixel_size=pixel_size_nm, units="nm")
    calculator.run_analysis(image.astype(np.float32))
    return float(calculator.resolution)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--dataset", choices=["BioSR_MT", "BioTISR_CCP"], default="BioSR_MT")
    parser.add_argument(
        "--runs-dir", type=Path, default=root / "experiments" / "prompt_ablation"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, runs_dir = args.repo_root.resolve(), args.runs_dir.resolve()
    reference_dir = {
        "BioSR_MT": root / "example" / "data" / "BioSR_MT" / "test" / "channel_0" / "SIM",
        "BioTISR_CCP": root / "example" / "data" / "BioTISR_CCP" / "train" / "channel_0" / "SIM",
    }[args.dataset]
    pixel_size_nm = {"BioSR_MT": 31.3, "BioTISR_CCP": 31.3}[args.dataset]
    output_conditions = list((runs_dir / "outputs").glob("[0-9][0-9]_*")) if (runs_dir / "outputs").is_dir() else []
    conditions_root = runs_dir / "outputs" if len(output_conditions) == 4 else runs_dir
    conditions = sorted(path for path in conditions_root.glob("[0-9][0-9]_*") if path.is_dir())
    if not conditions:
        raise FileNotFoundError(f"No condition directories found in {runs_dir}")

    rows: list[dict[str, object]] = []
    for condition in conditions:
        for prediction_path in sorted(condition.glob("*.tif")):
            reference_path = reference_dir / prediction_path.name
            if not reference_path.exists():
                raise FileNotFoundError(f"Reference image missing: {reference_path}")
            # BioSR_MT TIFFs are stored as single-plane stacks ``(1, H, W)``.
            # SSIM interprets a 3-D array as volumetric data, so remove only
            # singleton axes before comparing the two grayscale images.
            reference = normalize(tifffile.imread(reference_path)).squeeze()
            prediction = normalize(tifffile.imread(prediction_path)).squeeze()
            if reference.shape != prediction.shape:
                raise ValueError(
                    f"Shape mismatch for {prediction_path.name}: prediction {prediction.shape}, reference {reference.shape}. "
                    "Check that the inference scale factor and reference image are compatible."
                )
            rse, rsp = squirrel(reference, prediction)
            msssim = float(ms_ssim(torch.from_numpy(reference)[None, None], torch.from_numpy(prediction)[None, None], data_range=1.0).item())
            rows.append(
                {
                    "condition": condition.name,
                    "image": prediction_path.name,
                    "psnr": peak_signal_noise_ratio(reference, prediction, data_range=1.0),
                    "ssim": structural_similarity(reference, prediction, data_range=1.0),
                    "msssim": msssim,
                    "zncc": zncc(reference, prediction),
                    "nrmse": normalized_root_mse(reference, prediction),
                    "rse": rse,
                    "rsp": rsp,
                    "resolution_da_nm": resolution_da(prediction, pixel_size_nm),
                }
            )

    metrics_path = runs_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "image", "psnr", "ssim", "msssim", "zncc", "nrmse", "rse", "rsp", "resolution_da_nm"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = runs_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "n", "psnr_mean", "ssim_mean", "msssim_mean", "zncc_mean", "nrmse_mean", "rse_mean", "rsp_mean", "resolution_da_nm_mean"])
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
                    "rse_mean": np.mean([float(row["rse"]) for row in subset]),
                    "rsp_mean": np.mean([float(row["rsp"]) for row in subset]),
                    "resolution_da_nm_mean": np.mean([float(row["resolution_da_nm"]) for row in subset]),
                }
            )
    print(f"Wrote {metrics_path} and {summary_path}")


if __name__ == "__main__":
    main()
