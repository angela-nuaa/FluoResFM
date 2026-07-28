"""Render a qualitative BioSR_MT prompt-ablation panel without changing inputs.

The figure is a visual check for the screening metrics, not a paper-protocol
evaluation.  It reads the transferred experiment directory by default and
writes a new PNG there.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.transform import resize


CONDITIONS = (
    ("Full prompt", "01_full"),
    ("Task only", "02_task_only"),
    ("Wrong structure", "03_wrong_structure"),
    ("Wrong task", "04_wrong_task"),
)


def display_image(path: Path, output_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read a singleton-plane TIFF and percentile-normalise it for display."""
    image = tifffile.imread(path).squeeze().astype(np.float32)
    if output_shape is not None and image.shape != output_shape:
        image = resize(image, output_shape, order=1, preserve_range=True, anti_aliasing=True)
    low, high = np.quantile(image, (0.003, 0.995))
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--dataset", choices=["BioSR_MT", "BioTISR_CCP"], default="BioSR_MT")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220, help="PNG export resolution (default: 220).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, run_dir = args.repo_root.resolve(), args.run_dir.resolve()
    images = args.images or {
        "BioSR_MT": ["41.tif", "48.tif", "55.tif"],
        "BioTISR_CCP": ["Cell_001_0.tif", "Cell_001_10.tif", "Cell_001_19.tif"],
    }[args.dataset]
    # Cloud runs write conditions directly to ``run_dir``; transferred runs
    # preserve them under ``run_dir/outputs`` alongside provenance files.
    outputs_dir = run_dir / "outputs" if (run_dir / "outputs").is_dir() else run_dir
    output = args.output or run_dir / f"qualitative_panel_{args.dataset.lower()}.png"
    input_dir, reference_dir = {
        "BioSR_MT": (
            root / "example" / "data" / "BioSR_MT" / "test" / "channel_0" / "WF_noise_level_3",
            root / "example" / "data" / "BioSR_MT" / "test" / "channel_0" / "SIM",
        ),
        "BioTISR_CCP": (
            root / "example" / "data" / "BioTISR_CCP" / "train" / "channel_0" / "WF_noise_level_2",
            root / "example" / "data" / "BioTISR_CCP" / "train" / "channel_0" / "SIM",
        ),
    }[args.dataset]
    columns = (("Input (WF)", input_dir), ("Reference (SIM)", reference_dir), *CONDITIONS)

    figure, axes = plt.subplots(len(images), len(columns), figsize=(15, 7.5))
    figure.subplots_adjust(left=0.04, right=0.995, top=0.90, bottom=0.08, wspace=0.06, hspace=0.04)
    for row, image_name in enumerate(images):
        reference_path = reference_dir / image_name
        reference = display_image(reference_path)
        for column, (label, directory) in enumerate(columns):
            if directory in (input_dir, reference_dir):
                path = directory / image_name
            else:
                path = outputs_dir / directory / image_name
            if not path.exists():
                raise FileNotFoundError(f"Missing image: {path}")
            image = display_image(path, reference.shape)
            axis = axes[row, column]
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_axis_off()
            if row == 0:
                axis.set_title(label)
            if column == 0:
                axis.text(-0.12, 0.5, image_name, transform=axis.transAxes, rotation=90, ha="center", va="center")

    split = "training samples" if args.dataset == "BioTISR_CCP" else "test samples"
    figure.suptitle(f"{args.dataset} prompt ablation: qualitative check ({split})", fontsize=14)
    figure.text(0.5, 0.01, "Each panel is independently percentile-normalised for viewing; do not use this figure for quantitative comparison.", ha="center", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
