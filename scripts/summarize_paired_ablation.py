"""Summarise paired prompt-ablation metric changes against the full prompt."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="Metric CSV produced by an ablation evaluator")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference-condition", default="01_full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidate_fields = ("psnr", "ssim", "msssim", "zncc", "nrmse", "rse", "rsp", "resolution_da_nm")
    fields = tuple(field for field in candidate_fields if field in rows[0])
    if not fields:
        raise ValueError("Metrics CSV has no recognised metric columns")
    by_key = {(row["condition"], row["image"]): row for row in rows}
    images = sorted({row["image"] for row in rows if row["condition"] == args.reference_condition})
    conditions = sorted({row["condition"] for row in rows if row["condition"] != args.reference_condition})
    output_rows: list[dict[str, object]] = []
    comparisons = len(conditions) * len(fields)
    for condition in conditions:
        for metric in fields:
            deltas = np.asarray(
                [float(by_key[(condition, image)][metric]) - float(by_key[(args.reference_condition, image)][metric]) for image in images]
            )
            _, p_value = wilcoxon(deltas, alternative="two-sided", method="auto")
            output_rows.append(
                {
                    "reference_condition": args.reference_condition,
                    "condition": condition,
                    "metric": metric,
                    "n": len(deltas),
                    "mean_condition_minus_reference": float(np.mean(deltas)),
                    "median_condition_minus_reference": float(np.median(deltas)),
                    "all_deltas_same_direction": bool(np.all(deltas < 0) or np.all(deltas > 0)),
                    "wilcoxon_p_two_sided": float(p_value),
                    "bonferroni_p": min(float(p_value) * comparisons, 1.0),
                    "comparisons": comparisons,
                }
            )
    output = args.output or args.metrics.with_name("paired_summary.csv")
    fieldnames = list(output_rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
