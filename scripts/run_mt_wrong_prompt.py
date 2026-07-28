"""Run wrong-prompt control for BioSR_MT.

Three conditions (task / structure / imaging) each replace one field with a
plausible but incorrect value, keeping all other text verbatim from the
canonical wf_to_sim_2x prompt.  01_full is NOT re-generated — this script
only produces 05_wrong_task, 06_wrong_structure, 07_wrong_imaging.

Requires the completed P1 baseline under a separate runs-dir so that
01_full TIFFs can be linked/copied during downstream evaluation.
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
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# Canonical prompt is built from the wf_to_sim_2x protocol, matching
# example/data/text/train/dataset_text_ALL.txt line 1.
PROMPT_FULL = (
    "Task: super-resolution with a scale factor of 2; "
    "sample: fixed COS-7 cell line; "
    "structure: microtubule; "
    "fluorescence indicator: mEmerald (GFP); "
    "input microscope: wide-field microscope with excitation numerical aperture (NA) of 1.35, "
    "detection numerical aperture (NA) of 1.3; "
    "input pixel size: 62.6 x 62.6 nm. Nearest interpolation with a factor of 2; "
    "target microscope: linear structured illumination microscope with excitation numerical aperture (NA) of 1.35, "
    "detection numerical aperture (NA) of 1.3; "
    "target pixel size: 31.3 x 31.3 nm."
)

CONDITIONS = {
    "05_wrong_task": (
        "Task: deconvolution; "
        "sample: fixed COS-7 cell line; "
        "structure: microtubule; "
        "fluorescence indicator: mEmerald (GFP); "
        "input microscope: wide-field microscope with excitation numerical aperture (NA) of 1.35, "
        "detection numerical aperture (NA) of 1.3; "
        "input pixel size: 62.6 x 62.6 nm. Nearest interpolation with a factor of 2; "
        "target microscope: linear structured illumination microscope with excitation numerical aperture (NA) of 1.35, "
        "detection numerical aperture (NA) of 1.3; "
        "target pixel size: 31.3 x 31.3 nm."
    ),
    "06_wrong_structure": (
        "Task: super-resolution with a scale factor of 2; "
        "sample: fixed COS-7 cell line; "
        "structure: mitochondria; "
        "fluorescence indicator: mEmerald (GFP); "
        "input microscope: wide-field microscope with excitation numerical aperture (NA) of 1.35, "
        "detection numerical aperture (NA) of 1.3; "
        "input pixel size: 62.6 x 62.6 nm. Nearest interpolation with a factor of 2; "
        "target microscope: linear structured illumination microscope with excitation numerical aperture (NA) of 1.35, "
        "detection numerical aperture (NA) of 1.3; "
        "target pixel size: 31.3 x 31.3 nm."
    ),
    "07_wrong_imaging": (
        "Task: super-resolution with a scale factor of 2; "
        "sample: fixed COS-7 cell line; "
        "structure: microtubule; "
        "fluorescence indicator: mEmerald (GFP); "
        "input microscope: wide-field microscope with excitation numerical aperture (NA) of 1.35, "
        "detection numerical aperture (NA) of 1.3; "
        "input pixel size: 62.6 x 62.6 nm. Nearest interpolation with a factor of 2; "
        "target microscope: wide-field microscope with excitation numerical aperture (NA) of 1.35, "
        "detection numerical aperture (NA) of 1.3; "
        "target pixel size: 31.3 x 31.3 nm."
    ),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, runs_dir = args.repo_root.resolve(), args.runs_dir.resolve()
    plugin_source = root / "repos" / "napari-fluoresfm" / "src"
    if not plugin_source.is_dir():
        raise FileNotFoundError(f"Plugin source not found: {plugin_source}")
    sys.path.insert(0, str(plugin_source))
    from napari_fluoresfm.fluoresfm.test.predict import predict

    # Validate canonical prompt against the bundled training text
    data_root = root / "example" / "data" / "BioSR_MT"
    canonical_path = root / "example" / "data" / "text" / "train" / "dataset_text_ALL.txt"
    canonical_line = canonical_path.read_text(encoding="utf-8").splitlines()[0]
    if PROMPT_FULL != canonical_line:
        raise RuntimeError(
            f"PROMPT_FULL does not match {canonical_path} line 1.\n"
            f"  PROMPT_FULL: {PROMPT_FULL}\n"
            f"  CANONICAL:   {canonical_line}"
        )

    source_index = data_root / "test.txt"
    images = source_index.read_text(encoding="utf-8").splitlines()
    print(f"Found {len(images)} images in {source_index}")

    common = {
        "path_input": data_root / "test" / "channel_0" / "WF_noise_level_3",
        "path_input_index": source_index,
        "path_embedder": root / "example" / "checkpoints" / "biomedclip",
        "path_checkpoint": root / "example" / "checkpoints" / "fluoresfm" / "epoch_0_iter_700000.pt",
        "sf_lr": 2,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "device": args.device,
        "compile": False,
    }
    for name, value in common.items():
        if name.startswith("path_") and not Path(value).exists():
            raise FileNotFoundError(f"Required asset missing ({name}): {value}")

    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runs_dir / "manifest.json"
    existing_conditions: dict[str, str] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_conditions = previous.get("conditions", {})

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "BioSR_MT wrong-prompt control",
        "prompt_source": "example/data/text/train/dataset_text_ALL.txt line 1 (canonical full). "
        "Each condition replaces exactly one semantic field with an incorrect value.",
        "error_conditions": {
            "05_wrong_task": "Task → deconvolution (conflicting with sf_lr=2 which stays 2x output scale). "
            "All other fields verbatim from canonical.",
            "06_wrong_structure": "structure → mitochondria (microtubule → mitochondria). "
            "All other fields verbatim from canonical.",
            "07_wrong_imaging": "target microscope → wide-field microscope (SIM → wide-field). "
            "NA values unchanged; not editing task or scale factor.",
        },
        "root_commit": git_revision(root),
        "napari_fluoresfm_commit": git_revision(root / "repos" / "napari-fluoresfm"),
        "fluoresfm_commit": git_revision(root / "repos" / "fluoresfm"),
        "parameters": {key: str(value) for key, value in common.items()},
        "conditions": {**existing_conditions, **CONDITIONS},
        "baseline_dir": "experiments/mt_field_ablation/20260728_canonical_text_full15",
        "scope": "15 test images × 3 wrong-prompt conditions = 45 predictions. "
        "Comparison against the 01_full condition from the P1 baseline run.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for name, text in CONDITIONS.items():
        output = runs_dir / name
        if output.exists():
            if not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {output}; use --overwrite.")
            shutil.rmtree(output)
        output.mkdir(parents=True)
        params = {key: str(value) if isinstance(value, Path) else value for key, value in common.items()}
        params.update({"path_output": str(output), "text": text})
        print(f"\n=== {name} ===\n{text}\n")
        if predict(params) != 1:
            raise RuntimeError(f"Prediction failed for {name}")

    print(f"\nDone.  {len(CONDITIONS)} conditions written to {runs_dir}")


if __name__ == "__main__":
    main()
