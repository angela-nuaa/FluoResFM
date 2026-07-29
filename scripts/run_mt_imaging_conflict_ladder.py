"""Run P2-2 target-imaging metadata conflict controls for bundled BioSR_MT.

Generates only the three erroneous-prompt conditions.  The canonical full
prompt is deliberately not regenerated: downstream evaluation must use the
``01_full`` outputs from a separately recorded strict P2-1 baseline run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_provenance import provenance, write_run_md


FULL = (
    "Task: super-resolution with a scale factor of 2; sample: fixed COS-7 cell line; "
    "structure: microtubule; fluorescence indicator: mEmerald (GFP); "
    "input microscope: wide-field microscope with excitation numerical aperture (NA) of 1.35, "
    "detection numerical aperture (NA) of 1.3; "
    "input pixel size: 62.6 x 62.6 nm. Nearest interpolation with a factor of 2; "
    "target microscope: linear structured illumination microscope with excitation numerical aperture (NA) of 1.35, "
    "detection numerical aperture (NA) of 1.3; target pixel size: 31.3 x 31.3 nm."
)

CONDITIONS = {
    "02_wrong_target_modality": FULL.replace(
        "target microscope: linear structured illumination microscope",
        "target microscope: wide-field microscope",
    ),
    "03_wrong_target_pixel_size": FULL.replace(
        "target pixel size: 31.3 x 31.3 nm", "target pixel size: 62.6 x 62.6 nm"
    ),
    "04_wrong_target_modality_and_pixel_size": FULL.replace(
        "target microscope: linear structured illumination microscope",
        "target microscope: wide-field microscope",
    ).replace("target pixel size: 31.3 x 31.3 nm", "target pixel size: 62.6 x 62.6 nm"),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--smoke-test", action="store_true", help="Use only the first two indexed images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, runs_dir = args.repo_root.resolve(), args.runs_dir.resolve()
    if runs_dir.exists():
        raise FileExistsError(f"Run directory already exists: {runs_dir}")
    canonical_path = root / "example" / "data" / "text" / "train" / "dataset_text_ALL.txt"
    if canonical_path.read_text(encoding="utf-8").splitlines()[0] != FULL:
        raise RuntimeError("FULL must exactly match the bundled canonical prompt")
    plugin_source = root / "repos" / "napari-fluoresfm" / "src"
    sys.path.insert(0, str(plugin_source))
    from napari_fluoresfm.fluoresfm.test.predict import predict

    data_root = root / "example" / "data" / "BioSR_MT"
    source_index = data_root / "test.txt"
    images = [line for line in source_index.read_text(encoding="utf-8").splitlines() if line]
    if args.smoke_test:
        images = images[:2]
    runs_dir.mkdir(parents=True)
    input_index = runs_dir / "input_index.txt"
    input_index.write_text("\n".join(images) + "\n", encoding="utf-8")
    common = {
        "path_input": data_root / "test" / "channel_0" / "WF_noise_level_3",
        "path_input_index": input_index,
        "path_embedder": root / "example" / "checkpoints" / "biomedclip",
        "path_checkpoint": root / "example" / "checkpoints" / "fluoresfm" / "epoch_0_iter_700000.pt",
        "sf_lr": 2, "batch_size": args.batch_size, "patch_size": args.patch_size,
        "device": args.device, "compile": False,
    }
    for name, value in common.items():
        if name.startswith("path_") and not Path(value).exists():
            raise FileNotFoundError(f"Required asset missing ({name}): {value}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "P2-2 target-imaging metadata conflict ladder (BioSR_MT)",
        "evidence_level": "P2",
        "baseline_run": "experiments/mt_semantic_paraphrase/20260728_p2-1_strict_full15_r1/01_full",
        "scope": f"{'Smoke test (2 images)' if args.smoke_test else '15 images'} × 3 erroneous target-metadata conditions; not a generalisation claim.",
        "conditions": CONDITIONS,
        "condition_notes": {
            "02_wrong_target_modality": "Only target modality is changed from linear structured illumination to wide-field.",
            "03_wrong_target_pixel_size": "Only target pixel size is changed from 31.3 to 62.6 nm; the 2x task and all other text remain fixed.",
            "04_wrong_target_modality_and_pixel_size": "Both target modality and target pixel size are changed together.",
        },
        "parameters": {key: str(value) for key, value in common.items()},
        "provenance": provenance(root=root, script=Path(__file__), assets={"source_index": source_index, "input_index": input_index, "checkpoint": common["path_checkpoint"], "canonical_prompt": canonical_path}, parameters=common),
    }
    (runs_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_run_md(runs_dir / "run.md", manifest)
    for name, text in CONDITIONS.items():
        output = runs_dir / name
        output.mkdir()
        params = {key: str(value) if isinstance(value, Path) else value for key, value in common.items()}
        params.update({"path_output": str(output), "text": text})
        print(f"\n=== {name} ===\n{text}\n")
        if predict(params) != 1:
            raise RuntimeError(f"Prediction failed for {name}")


if __name__ == "__main__":
    main()
