"""Evaluate prediction disagreement as a P2 prompt-risk signal on BioSR_MT.

The script never selects or blends candidate predictions.  It measures how far
each candidate-prompt output moves from the canonical output, then reports its
descriptive association with the paired reference-fidelity change.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity


def normalize(image: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(image.astype(np.float32), (0.03, 0.995))
    if not hi > lo:
        raise ValueError("Image has insufficient intensity range")
    return np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_metrics(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"condition", "image", "psnr", "ssim", "msssim", "zncc", "nrmse", "rse", "rsp"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("metrics CSV is incomplete")
    return {(row["condition"], row["image"]): row for row in rows}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs_dir, output = args.runs_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    metrics = read_metrics(args.metrics.resolve())
    baseline_dir = runs_dir / "01_full"
    conditions = sorted(path.name for path in runs_dir.glob("[0-9][0-9]_*") if path.is_dir() and path.name != "01_full")
    images = sorted(path.name for path in baseline_dir.glob("*.tif"))
    if len(images) != 15:
        raise ValueError(f"Expected 15 baseline TIFFs, found {len(images)}")
    rows: list[dict[str, object]] = []
    for condition in conditions:
        for image in images:
            baseline = normalize(tifffile.imread(baseline_dir / image)).squeeze()
            candidate = normalize(tifffile.imread(runs_dir / condition / image)).squeeze()
            if baseline.shape != candidate.shape:
                raise ValueError(f"Shape mismatch for {condition}/{image}")
            base_metric, candidate_metric = metrics[("01_full", image)], metrics[(condition, image)]
            rows.append(
                {
                    "condition": condition,
                    "image": image,
                    "prediction_mae": float(np.mean(np.abs(candidate - baseline))),
                    "prediction_rmse": float(np.sqrt(np.mean((candidate - baseline) ** 2))),
                    "prediction_ssim": float(structural_similarity(baseline, candidate, data_range=1.0)),
                    "delta_psnr": float(candidate_metric["psnr"]) - float(base_metric["psnr"]),
                    "delta_ssim": float(candidate_metric["ssim"]) - float(base_metric["ssim"]),
                    "delta_msssim": float(candidate_metric["msssim"]) - float(base_metric["msssim"]),
                    "delta_zncc": float(candidate_metric["zncc"]) - float(base_metric["zncc"]),
                    "delta_nrmse": float(candidate_metric["nrmse"]) - float(base_metric["nrmse"]),
                    "delta_rse": float(candidate_metric["rse"]) - float(base_metric["rse"]),
                    "delta_rsp": float(candidate_metric["rsp"]) - float(base_metric["rsp"]),
                }
            )
    write_csv(output / "per_image_prompt_disagreement.csv", rows)
    summaries: list[dict[str, object]] = []
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        for disagreement in ("prediction_mae", "prediction_rmse", "prediction_ssim"):
            values = np.asarray([float(row[disagreement]) for row in subset])
            if disagreement == "prediction_ssim":
                values = -values  # larger value means more disagreement in every summary row
            for fidelity in ("delta_psnr", "delta_ssim", "delta_msssim", "delta_zncc", "delta_nrmse", "delta_rse", "delta_rsp"):
                fidelity_values = np.asarray([float(row[fidelity]) for row in subset])
                if fidelity in {"delta_nrmse", "delta_rse"}:
                    fidelity_values = -fidelity_values
                rho, p_value = spearmanr(values, -fidelity_values)  # positive: more disagreement accompanies more loss
                summaries.append({"condition": condition, "disagreement": disagreement, "fidelity": fidelity, "n": len(subset), "spearman_rho_more_disagreement_vs_more_loss": rho, "p_two_sided": p_value})
    write_csv(output / "within_condition_spearman.csv", summaries)
    print(f"Wrote {output} ({len(rows)} prediction pairs)")


if __name__ == "__main__":
    main()
