"""Evaluate each prompt-ablation TIFF in a fresh process and merge all metrics.

NanoPyx can accumulate resources when RSE/RSP/decorrelation analysis is run on
many images in one Python process.  This wrapper calls the existing evaluator
once per image, so a completed image remains valid even if a later image fails.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--dataset", choices=["BioSR_MT", "BioTISR_CCP"], default="BioSR_MT")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of isolated image evaluators to run concurrently (default: 8).",
    )
    return parser.parse_args()


def evaluate_one(
    evaluator: Path,
    workspace: Path,
    dataset: str,
    condition: Path,
    prediction: Path,
) -> dict[str, str]:
    """Evaluate one image in a fresh process and return its sole metric row."""
    job = workspace / condition.name / prediction.stem
    linked_condition = job / condition.name
    linked_condition.mkdir(parents=True, exist_ok=True)
    linked_image = linked_condition / prediction.name
    if not linked_image.exists():
        # Windows commonly disallows unprivileged symbolic links.  A hard link
        # keeps the one-image evaluator lightweight there; copying is the
        # portable last resort (for example, across volumes).
        try:
            linked_image.hardlink_to(prediction)
        except OSError:
            shutil.copy2(prediction, linked_image)

    # Each subprocess is independent, so this retains the resource-isolation
    # safeguard while several images can use separate CPU cores.  Limiting
    # native thread pools prevents a few workers from oversubscribing the CPU.
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    subprocess.run(
        [sys.executable, str(evaluator), "--dataset", dataset, "--runs-dir", str(job)],
        check=True,
        env=env,
    )
    with (job / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        image_rows = list(csv.DictReader(handle))
    if len(image_rows) != 1:
        key = f"{condition.name}/{prediction.name}"
        raise RuntimeError(f"Expected exactly one metric row for {key}, got {len(image_rows)}")
    return image_rows[0]


def write_rows(output: Path, rows: list[dict[str, str]]) -> None:
    ordered = sorted(rows, key=lambda row: (row["condition"], row["image"]))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)


def write_summary(metrics_output: Path, rows: list[dict[str, str]]) -> Path:
    """Write condition means alongside the resumable per-image CSV."""
    summary_output = metrics_output.with_name(f"{metrics_output.stem}_summary.csv")
    metric_fields = [field for field in rows[0] if field not in {"condition", "image"}]
    conditions = sorted({row["condition"] for row in rows})
    with summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "n", *[f"{field}_mean" for field in metric_fields]])
        writer.writeheader()
        for condition in conditions:
            subset = [row for row in rows if row["condition"] == condition]
            writer.writerow(
                {
                    "condition": condition,
                    "n": len(subset),
                    **{f"{field}_mean": fmean(float(row[field]) for row in subset) for field in metric_fields},
                }
            )
    return summary_output


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    root, runs_dir = args.repo_root.resolve(), args.runs_dir.resolve()
    output = args.output or runs_dir / "isolated_full_metrics.csv"
    evaluator = root / "scripts" / "evaluate_prompt_ablation.py"
    conditions = sorted(path for path in runs_dir.glob("[0-9][0-9]_*") if path.is_dir())
    if not conditions:
        raise FileNotFoundError(f"No condition directories found in {runs_dir}")
    workspace = runs_dir / "nanopyx_isolated"
    rows: list[dict[str, str]] = []
    if output.exists():
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    completed = {(row["condition"], row["image"]) for row in rows}

    pending: list[tuple[Path, Path]] = []
    for condition in conditions:
        for prediction in sorted(condition.glob("*.tif")):
            key = (condition.name, prediction.name)
            if key in completed:
                continue
            pending.append((condition, prediction))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_one, evaluator, workspace, args.dataset, condition, prediction): (condition, prediction)
            for condition, prediction in pending
        }
        for future in as_completed(futures):
            condition, prediction = futures[future]
            rows.append(future.result())
            completed.add((condition.name, prediction.name))
            write_rows(output, rows)
            print(f"Completed {condition.name}/{prediction.name} ({len(rows)} total)", flush=True)

    summary_output = write_summary(output, rows)
    print(f"Wrote {output} and {summary_output}")


if __name__ == "__main__":
    main()
