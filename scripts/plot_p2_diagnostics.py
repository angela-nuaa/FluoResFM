"""Render P2-4/P2-5 diagnostic figures from existing local CSV files.

This script is read-only with respect to experiment artifacts: it only consumes
previously generated paired CSVs and writes documentation PNGs.  The figures
are descriptive diagnostics on the bundled 15-image BioSR-MT setting, not
independent-test or generalisation evidence.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"p2_1": "#4C78A8", "p2_2": "#F58518", "p2_3": "#54A24B"}
EXPERIMENT_LABELS = {"p2_1": "P2-1", "p2_2": "P2-2", "p2_3": "P2-3"}
CONDITION_LABELS = {
    "02_wrong_input_modality": "Wrong input modality",
    "03_wrong_structure": "Wrong structure",
    "04_wrong_fluorescence_indicator": "Wrong fluorescence label",
}
CONDITION_COLORS = {
    "02_wrong_input_modality": "#4C78A8",
    "03_wrong_structure": "#F58518",
    "04_wrong_fluorescence_indicator": "#54A24B",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def plot_metric_direction(rows: list[dict[str, str]], output: Path) -> None:
    needed = {"experiment", "reference_fidelity_z_mean", "da_signed_change", "direction_disagrees"}
    if not needed.issubset(rows[0]):
        raise ValueError("P2-4 CSV has missing columns")

    figure, axis = plt.subplots(figsize=(7.1, 5.4), constrained_layout=True)
    for experiment in ("p2_1", "p2_2", "p2_3"):
        subset = [row for row in rows if row["experiment"] == experiment]
        same = [row for row in subset if row["direction_disagrees"].lower() != "true"]
        opposite = [row for row in subset if row["direction_disagrees"].lower() == "true"]
        color = COLORS[experiment]
        axis.scatter(
            [float(row["reference_fidelity_z_mean"]) for row in same],
            [float(row["da_signed_change"]) for row in same],
            s=28, alpha=0.62, color=color, linewidths=0, label=EXPERIMENT_LABELS[experiment],
        )
        axis.scatter(
            [float(row["reference_fidelity_z_mean"]) for row in opposite],
            [float(row["da_signed_change"]) for row in opposite],
            s=35, alpha=0.82, color=color, marker="x", linewidths=1.25,
        )
    axis.axhline(0, color="0.35", linewidth=0.9)
    axis.axvline(0, color="0.35", linewidth=0.9)
    axis.set_xlabel("Reference-fidelity composite\n(standardised paired change; positive = better)")
    axis.set_ylabel("Signed DA change (nm; positive = lower DA)")
    axis.set_title("P2-4: DA and reference metrics can move in opposite directions")
    axis.text(0.02, 0.02, "× = opposite direction\nEach mark = one paired image-condition result", transform=axis.transAxes,
              fontsize=8.5, va="bottom", ha="left", color="0.25")
    axis.legend(frameon=False, title="Source comparison")
    axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_prompt_disagreement(rows: list[dict[str, str]], output: Path) -> None:
    needed = {"condition", "prediction_rmse", "delta_psnr"}
    if not needed.issubset(rows[0]):
        raise ValueError("P2-5 CSV has missing columns")

    figure, axis = plt.subplots(figsize=(7.1, 5.4), constrained_layout=True)
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        axis.scatter(
            [float(row["prediction_rmse"]) for row in subset],
            [-float(row["delta_psnr"]) for row in subset],
            s=34, alpha=0.75, color=CONDITION_COLORS[condition], linewidths=0,
            label=CONDITION_LABELS[condition],
        )
    axis.axhline(0, color="0.35", linewidth=0.9)
    axis.set_xlabel("Candidate-to-full prediction RMSE")
    axis.set_ylabel("PSNR loss vs full prompt (dB; positive = worse)")
    axis.set_title("P2-5: prediction disagreement is a descriptive review signal")
    axis.text(0.02, 0.02, "15 images per condition; no decision threshold is validated", transform=axis.transAxes,
              fontsize=8.5, va="bottom", ha="left", color="0.25")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-4-csv", type=Path, required=True)
    parser.add_argument("--p2-5-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    plot_metric_direction(read_csv(args.p2_4_csv.resolve()), output / "p2_4_metric_direction_conflict.png")
    plot_prompt_disagreement(read_csv(args.p2_5_csv.resolve()), output / "p2_5_prompt_disagreement.png")
    print(f"Wrote P2 diagnostic figures to {output}")


if __name__ == "__main__":
    main()
