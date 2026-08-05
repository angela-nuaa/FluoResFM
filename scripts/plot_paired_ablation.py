"""Plot paired per-image changes for a prompt-ablation metrics CSV.

The plot preserves image-level pairing with the full-prompt baseline.  For all
panels, negative values mean lower reference-based fidelity than the baseline.
It is intentionally preferred to a radar chart, whose normalisation can change
the apparent result when metrics have different scales and directions.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("psnr", "PSNR (dB)"),
    ("zncc", "ZNCC"),
    ("nrmse", "NRMSE"),
    ("rse", "RSE"),
)

LABELS = {
    "02_task_paraphrase": "Task paraphrase",
    "03_structure_paraphrase": "Structure paraphrase",
    "04_imaging_paraphrase": "Imaging paraphrase",
    "05_full_paraphrase": "All paraphrased",
    "02_wrong_target_modality": "Wrong target modality",
    "03_wrong_target_pixel_size": "Wrong target pixel size",
    "04_wrong_target_modality_and_pixel_size": "Both target fields wrong",
    "02_wrong_input_modality": "Wrong input modality",
    "03_wrong_structure": "Wrong structure",
    "04_wrong_fluorescence_indicator": "Wrong fluorescence label",
    "02_no_task": "No task",
    "03_no_structure": "No structure",
    "04_no_imaging": "No imaging",
    "05_wrong_task": "Wrong task",
    "06_wrong_structure": "Wrong structure",
    "07_wrong_imaging": "Wrong imaging",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="isolated_full_metrics.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--title", default=None)
    parser.add_argument("--conditions", nargs="+", default=None, help="Condition IDs, excluding 01_full")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"condition", "image", *(key for key, _ in METRICS)}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is not a complete isolated metrics CSV")
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics)
    by_condition: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]][row["image"]] = {
            metric: float(row[metric]) for metric, _ in METRICS
        }
    if "01_full" not in by_condition:
        raise ValueError("The metrics file must contain condition 01_full")

    conditions = args.conditions or sorted(key for key in by_condition if key != "01_full")
    if not conditions:
        raise ValueError("No comparison conditions were selected")
    missing = [condition for condition in conditions if condition not in by_condition]
    if missing:
        raise ValueError(f"Missing conditions: {', '.join(missing)}")

    baseline = by_condition["01_full"]
    figure, axes = plt.subplots(1, len(METRICS), figsize=(13.2, 4.5), constrained_layout=True)
    rng = np.random.default_rng(20260728)
    for axis, (metric, label) in zip(axes, METRICS, strict=True):
        values: list[np.ndarray] = []
        for condition in conditions:
            shared = sorted(set(baseline) & set(by_condition[condition]))
            if not shared:
                raise ValueError(f"No matching image names for {condition}")
            raw = np.array([by_condition[condition][image][metric] - baseline[image][metric] for image in shared])
            # Lower NRMSE/RSE is better; reverse their sign so every panel shares
            # the same interpretation: below zero means worse than full prompt.
            values.append(-raw if metric in {"nrmse", "rse"} else raw)

        positions = np.arange(len(conditions))
        for position, series in zip(positions, values, strict=True):
            jitter = rng.uniform(-0.10, 0.10, size=len(series))
            axis.scatter(np.full(len(series), position) + jitter, series, s=22, alpha=0.62, color="#4C78A8", linewidths=0)
            mean = float(np.mean(series))
            axis.scatter(position, mean, marker="D", s=35, color="#D55E00", zorder=3, label="Mean" if position == 0 else None)
            axis.text(position, mean, f"{mean:.3g}", ha="center", va="bottom" if mean >= 0 else "top", fontsize=8)

        axis.axhline(0, color="0.35", linewidth=1)
        axis.set_title(label)
        axis.set_xticks(positions, [LABELS.get(condition, condition) for condition in conditions], rotation=25, ha="right")
        axis.set_ylabel("Signed change vs full prompt\n(negative = worse)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.12, y=0.18)

    axes[0].legend(frameon=False, loc="best")
    if args.title:
        figure.suptitle(args.title, fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
