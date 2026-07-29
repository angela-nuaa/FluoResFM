"""Diagnose agreement between reference metrics and DA on paired MT prompts.

This is a P2 diagnostic, not a quality score.  It combines already-evaluated
predictions, writes every paired delta, and inspects edge/high-frequency errors
only for interpreting rows where DA and reference-metric directions disagree.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.stats import spearmanr
from skimage.filters import sobel


REFERENCE_METRICS = ("psnr", "ssim", "msssim", "zncc", "rsp", "nrmse", "rse")
ALL_METRICS = (*REFERENCE_METRICS, "resolution_da_nm")
LOWER_IS_BETTER = {"nrmse", "rse", "resolution_da_nm"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--experiment",
        nargs=3,
        action="append",
        metavar=("NAME", "METRICS_CSV", "RUNS_DIR"),
        required=True,
        help="Repeat: experiment name, paired metrics CSV, and TIFF runs directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"condition", "image", *ALL_METRICS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is missing required metrics")
    return rows


def normalize(image: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(image.astype(np.float32), (0.03, 0.995))
    if not hi > lo:
        raise ValueError("Image has insufficient intensity range")
    return np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def hf_ratio(image: np.ndarray) -> float:
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image))) ** 2
    height, width = image.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt(((yy - height / 2) / (height / 2)) ** 2 + ((xx - width / 2) / (width / 2)) ** 2)
    total = float(spectrum.sum())
    return float(spectrum[radius >= 0.5].sum() / total) if total else float("nan")


def image_diagnostics(reference_path: Path, baseline_path: Path, candidate_path: Path) -> dict[str, float]:
    reference = normalize(tifffile.imread(reference_path)).squeeze()
    baseline = normalize(tifffile.imread(baseline_path)).squeeze()
    candidate = normalize(tifffile.imread(candidate_path)).squeeze()
    ref_edge = float(np.mean(np.abs(sobel(reference))))
    base_edge_error = abs(float(np.mean(np.abs(sobel(baseline)))) - ref_edge)
    candidate_edge_error = abs(float(np.mean(np.abs(sobel(candidate)))) - ref_edge)
    ref_hf = hf_ratio(reference)
    base_hf_error = abs(hf_ratio(baseline) - ref_hf)
    candidate_hf_error = abs(hf_ratio(candidate) - ref_hf)
    return {
        "edge_error_baseline": base_edge_error,
        "edge_error_candidate": candidate_edge_error,
        "edge_error_change": candidate_edge_error - base_edge_error,
        "hf_ratio_reference": ref_hf,
        "hf_error_baseline": base_hf_error,
        "hf_error_candidate": candidate_hf_error,
        "hf_error_change": candidate_hf_error - base_hf_error,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root, output = args.repo_root.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    reference_dir = root / "example" / "data" / "BioSR_MT" / "test" / "channel_0" / "SIM"
    paired: list[dict[str, object]] = []
    run_dirs: dict[str, Path] = {}
    for name, metrics_text, runs_text in args.experiment:
        metrics, runs_dir = Path(metrics_text).resolve(), Path(runs_text).resolve()
        run_dirs[name] = runs_dir
        rows = read_metrics(metrics)
        baseline = {row["image"]: row for row in rows if row["condition"] == "01_full"}
        if len(baseline) != 15:
            raise ValueError(f"{name}: expected 15 baseline rows, found {len(baseline)}")
        for row in rows:
            if row["condition"] == "01_full":
                continue
            image = row["image"]
            if image not in baseline:
                raise ValueError(f"{name}: missing baseline for {image}")
            base = baseline[image]
            record: dict[str, object] = {"experiment": name, "condition": row["condition"], "image": image}
            for metric in ALL_METRICS:
                delta = float(row[metric]) - float(base[metric])
                record[f"delta_{metric}"] = delta
                record[f"signed_delta_{metric}"] = -delta if metric in LOWER_IS_BETTER else delta
            paired.append(record)
    for metric in REFERENCE_METRICS:
        values = np.asarray([float(row[f"signed_delta_{metric}"]) for row in paired])
        scale = float(values.std(ddof=0))
        for row, value in zip(paired, values, strict=True):
            row[f"z_{metric}"] = float(value / scale) if scale else 0.0
    for row in paired:
        row["reference_fidelity_z_mean"] = float(np.mean([float(row[f"z_{metric}"]) for metric in REFERENCE_METRICS]))
        row["da_signed_change"] = float(row["signed_delta_resolution_da_nm"])
        row["direction_disagrees"] = bool(
            float(row["reference_fidelity_z_mean"]) * float(row["da_signed_change"]) < 0
        )
        row["discordance_score"] = abs(float(row["reference_fidelity_z_mean"])) + abs(float(row["da_signed_change"]) / 5.0)
    write_csv(output / "paired_metric_deltas.csv", paired)

    condition_summary: list[dict[str, object]] = []
    for experiment, condition in sorted({(str(row["experiment"]), str(row["condition"])) for row in paired}):
        subset = [row for row in paired if row["experiment"] == experiment and row["condition"] == condition]
        condition_summary.append(
            {
                "experiment": experiment,
                "condition": condition,
                "n": len(subset),
                "direction_disagrees_n": sum(bool(row["direction_disagrees"]) for row in subset),
                "mean_reference_fidelity_z": float(np.mean([float(row["reference_fidelity_z_mean"]) for row in subset])),
                "mean_signed_DA_change_nm": float(np.mean([float(row["da_signed_change"]) for row in subset])),
                "mean_delta_psnr": float(np.mean([float(row["delta_psnr"]) for row in subset])),
            }
        )
    write_csv(output / "condition_direction_summary.csv", condition_summary)

    correlation_rows: list[dict[str, object]] = []
    for metric in (*REFERENCE_METRICS, "reference_fidelity_z_mean"):
        x = np.asarray([float(row[f"signed_delta_{metric}"]) if metric in REFERENCE_METRICS else float(row[metric]) for row in paired])
        y = np.asarray([float(row["da_signed_change"]) for row in paired])
        rho, p_value = spearmanr(x, y)
        correlation_rows.append({"metric_or_composite": metric, "n": len(paired), "spearman_rho_with_signed_DA": rho, "p_two_sided": p_value})
    write_csv(output / "metric_da_spearman.csv", correlation_rows)

    discordant = [row for row in paired if bool(row["direction_disagrees"])]
    discordant.sort(key=lambda row: float(row["discordance_score"]), reverse=True)
    diagnostics: list[dict[str, object]] = []
    for row in discordant[: args.top_n]:
        runs_dir = run_dirs[str(row["experiment"])]
        image, condition = str(row["image"]), str(row["condition"])
        diagnostic = image_diagnostics(reference_dir / image, runs_dir / "01_full" / image, runs_dir / condition / image)
        diagnostics.append({**row, **diagnostic})
    if diagnostics:
        write_csv(output / "discordant_samples_with_edge_frequency.csv", diagnostics)

    x = np.asarray([float(row["reference_fidelity_z_mean"]) for row in paired])
    y = np.asarray([float(row["da_signed_change"]) for row in paired])
    fig, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    for experiment in sorted({str(row["experiment"]) for row in paired}):
        mask = np.asarray([row["experiment"] == experiment for row in paired])
        axis.scatter(x[mask], y[mask], s=34, alpha=0.72, label=experiment)
    axis.axhline(0, color="0.45", linewidth=0.9)
    axis.axvline(0, color="0.45", linewidth=0.9)
    axis.set_xlabel("Reference-fidelity composite (standardised paired change)")
    axis.set_ylabel("Signed DA change (nm; positive = lower DA)")
    axis.set_title("P2 metric-direction diagnostic")
    axis.legend(frameon=False)
    fig.savefig(output / "reference_fidelity_vs_da.png", dpi=220)
    print(f"Wrote {output} ({len(paired)} paired rows; {len(discordant)} direction-discordant rows)")


if __name__ == "__main__":
    main()
