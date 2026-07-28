"""Run a four-condition FluoResFM text-prompt ablation on BioSR_MT.

Run from the repository root with the CUDA-enabled ``fluoresfm`` environment.
The script reads only the bundled example assets and writes a new, ignored
experiment directory. It will not overwrite a prior run unless --overwrite is
explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_revision(repo: Path) -> str | None:
    """Return the current Git revision without failing an experiment run."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def full_prompt(task: str, structure: str, dataset: str) -> str:
    """Format the prompt fields exposed by the napari prediction widget."""
    if dataset == "BioTISR_CCP":
        return (
            f"Task: {task} with a scale factor of 2; sample: fixed COS-7 cell line; "
            f"structure: {structure}; fluorescence indicator: mEmerald (GFP); input microscope: "
            "wide-field microscope with excitation numerical aperture (NA) of 1.41, detection "
            "numerical aperture (NA) of 1.3; input pixel size: 62.6 x 62.6 nm. Nearest interpolation "
            "with a factor of 2.; target microscope: linear structured illumination microscope with "
            "excitation numerical aperture (NA) of 1.41, detection numerical aperture (NA) of 1.3; "
            "target pixel size: 31.3 x 31.3 nm."
        )
    return (
        f"Task: {task}; sample: fixed COS-7 cell line; structure: {structure}; "
        "fluorescence indicator: mEmerald (GFP); input microscope: wide-field "
        "microscope with excitation numerical aperture (NA) of 1.35, detection "
        "numerical aperture (NA) of 1.3, wavelength of 488 nm; input pixel size: "
        "62.6 x 62.6 nm; target microscope: linear structured illumination "
        "microscope with excitation numerical aperture (NA) of 1.35, detection "
        "numerical aperture (NA) of 1.3, wavelength of 488 nm; target pixel size: "
        "31.3 x 31.3 nm."
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--dataset", choices=["BioSR_MT", "BioTISR_CCP"], default="BioSR_MT")
    parser.add_argument(
        "--runs-dir", type=Path, default=repo_root / "experiments" / "prompt_ablation"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument(
        "--images",
        nargs="+",
        metavar="FILE",
        help="Optional subset of dataset filenames (for example: 41.tif).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["full", "task_only", "wrong_structure", "wrong_task"],
        help="Optional prompt-condition subset; defaults to all four conditions.",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=2,
        help="Nearest-neighbour input interpolation; 2 matches 62.6 nm input to 31.3 nm SIM reference.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    plugin_source = root / "repos" / "napari-fluoresfm" / "src"
    if not plugin_source.is_dir():
        raise FileNotFoundError(f"Plugin source not found: {plugin_source}")
    sys.path.insert(0, str(plugin_source))

    from napari_fluoresfm.fluoresfm.test.predict import predict

    dataset_config = {
        "BioSR_MT": {
            "input_root": root / "example" / "data" / "BioSR_MT" / "test" / "channel_0",
            "input_dir": "WF_noise_level_3",
            "index": root / "example" / "data" / "BioSR_MT" / "test.txt",
            "structure": "microtubules",
            "wrong_structure": "mitochondria",
        },
        "BioTISR_CCP": {
            "input_root": root / "example" / "data" / "BioTISR_CCP" / "train" / "channel_0",
            "input_dir": "WF_noise_level_2",
            "index": root / "example" / "data" / "BioTISR_CCP" / "train_2d.txt",
            "structure": "clathrin-coated pits",
            "wrong_structure": "microtubules",
        },
    }[args.dataset]
    input_root = dataset_config["input_root"]
    runs_dir = args.runs_dir.resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    source_index = dataset_config["index"]
    input_index = source_index
    if args.images:
        available = set(source_index.read_text(encoding="utf-8").splitlines())
        missing = set(args.images) - available
        if missing:
            raise ValueError(f"Image(s) not in {source_index}: {sorted(missing)}")
        input_index = runs_dir / "input_index.txt"
        input_index.write_text("\n".join(args.images) + "\n", encoding="utf-8")
    common = {
        "path_input": input_root / dataset_config["input_dir"],
        "path_input_index": input_index,
        "path_embedder": root / "example" / "checkpoints" / "biomedclip",
        "path_checkpoint": root
        / "example"
        / "checkpoints"
        / "fluoresfm"
        / "epoch_0_iter_700000.pt",
        "sf_lr": args.scale_factor,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "device": args.device,
        "compile": False,
    }
    for name, path in common.items():
        if name.startswith("path_") and not Path(path).exists():
            raise FileNotFoundError(f"Required asset missing ({name}): {path}")

    # The task label follows the WF (62.6 nm) -> SIM (31.3 nm) scale change.
    # For BioSR_MT it is an experimental assumption; BioTISR_CCP uses the
    # upstream napari plugin's CCP example prompt fields.
    all_conditions = {
        "01_full": full_prompt("super-resolution", dataset_config["structure"], args.dataset),
        "02_task_only": "Task: super-resolution.",
        "03_wrong_structure": full_prompt("super-resolution", dataset_config["wrong_structure"], args.dataset),
        "04_wrong_task": full_prompt("deconvolution", dataset_config["structure"], args.dataset),
    }
    condition_keys = {
        "full": "01_full",
        "task_only": "02_task_only",
        "wrong_structure": "03_wrong_structure",
        "wrong_task": "04_wrong_task",
    }
    conditions = {
        condition_keys[name]: all_conditions[condition_keys[name]]
        for name in (args.conditions or condition_keys)
    }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root_commit": git_revision(root),
        "napari_fluoresfm_commit": git_revision(root / "repos" / "napari-fluoresfm"),
        "fluoresfm_commit": git_revision(root / "repos" / "fluoresfm"),
        "parameters": {key: str(value) for key, value in common.items()},
        "conditions": conditions,
        "dataset": args.dataset,
        "note": "The task label follows the example's 2x pixel-size change; BioSR_MT prompt metadata are an inference, while BioTISR_CCP uses the upstream CCP example prompt.",
    }
    (runs_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for name, prompt in conditions.items():
        output = runs_dir / name
        if output.exists():
            if not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {output}; use --overwrite.")
            shutil.rmtree(output)
        output.mkdir(parents=True)
        params = {key: str(value) if isinstance(value, Path) else value for key, value in common.items()}
        params.update({"path_output": str(output), "text": prompt})
        print(f"\n=== {name} ===\n{prompt}\n")
        if predict(params) != 1:
            raise RuntimeError(f"Prediction failed for {name}")


if __name__ == "__main__":
    main()
