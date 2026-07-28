"""Run a single-field prompt ablation on bundled BioTISR_CCP training samples.

This mirrors the controlled BioSR_MT field-deletion design: starting from the
BioTISR_CCP super-resolution row in ``example/data/finetune.xlsx``, omit
exactly one of task, structure, or imaging metadata.  BioTISR_CCP has no
bundled held-out split, so the resulting numbers are a within-directory
prompt-sensitivity screen, not generalisation evidence.
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
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prompt(*, task: bool = True, structure: bool = True, imaging: bool = True) -> str:
    """Build CCP prompts by omitting exactly one semantic field when requested."""
    fields: list[str] = []
    if task:
        fields.append("Task: super-resolution with a scale factor of 2")
    fields.append("sample: live SUM-159 cell")
    if structure:
        fields.append("structure: clathrin-coated pits")
    fields.append("fluorescence indicator: mEmerald (GFP)")
    if imaging:
        fields.extend(
            [
                "input microscope: wide-field microscope with excitation numerical aperture (NA) of 1.41.",
                "input pixel size: 60.4 x 60.4 nm. Nearest interpolation with a factor of 2",
                "target microscope: linear structured illumination microscope with excitation numerical aperture (NA) of 1.41.",
                "target pixel size: 30.2 x 30.2 nm",
            ]
        )
    return "; ".join(fields) + "."


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--images", nargs="+", metavar="FILE")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["full", "no_task", "no_structure", "no_imaging"],
        help="Defaults to all four conditions.",
    )
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

    data_root = root / "example" / "data" / "BioTISR_CCP"
    source_index = data_root / "train_2d.txt"
    input_index = source_index
    if args.images:
        available = set(source_index.read_text(encoding="utf-8").splitlines())
        missing = set(args.images) - available
        if missing:
            raise ValueError(f"Image(s) not in {source_index}: {sorted(missing)}")
        runs_dir.mkdir(parents=True, exist_ok=True)
        input_index = runs_dir / "input_index.txt"
        input_index.write_text("\n".join(args.images) + "\n", encoding="utf-8")

    common = {
        "path_input": data_root / "train" / "channel_0" / "WF_noise_level_2",
        "path_input_index": input_index,
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

    all_conditions = {
        "01_full": prompt(),
        "02_no_task": prompt(task=False),
        "03_no_structure": prompt(structure=False),
        "04_no_imaging": prompt(imaging=False),
    }
    condition_keys = {
        "full": "01_full",
        "no_task": "02_no_task",
        "no_structure": "03_no_structure",
        "no_imaging": "04_no_imaging",
    }
    selected = args.conditions or ("full", "no_task", "no_structure", "no_imaging")
    conditions = {condition_keys[name]: all_conditions[condition_keys[name]] for name in selected}

    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runs_dir / "manifest.json"
    existing_conditions: dict[str, str] = {}
    if manifest_path.exists():
        existing_conditions = json.loads(manifest_path.read_text(encoding="utf-8")).get("conditions", {})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "BioTISR_CCP",
        "root_commit": git_revision(root),
        "napari_fluoresfm_commit": git_revision(root / "repos" / "napari-fluoresfm"),
        "fluoresfm_commit": git_revision(root / "repos" / "fluoresfm"),
        "parameters": {key: str(value) for key, value in common.items()},
        "conditions": {**existing_conditions, **conditions},
        "scope": "Controlled prompt screen on bundled BioTISR_CCP training-directory samples; not a held-out evaluation.",
        "prompt_source": "Task and field values follow the BioTISR_CCP super-resolution row in example/data/finetune.xlsx; each non-full condition deletes one semantic field.",
        "input_variant_note": "WF_noise_level_2 is retained for comparability with the prior CCP screen. The workbook row names cropped WF_noise_level_0_0 training patches, so this is not an exact path-level reproduction of that row.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for name, text in conditions.items():
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


if __name__ == "__main__":
    main()
