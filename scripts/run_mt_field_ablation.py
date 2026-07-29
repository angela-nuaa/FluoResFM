"""Run a single-field text ablation on BioSR_MT test samples.

Two explicit protocols are available. ``upstream_deconv_1x`` follows the
upstream prediction example and is qualitative because its output has no
same-size bundled reference. ``wf_to_sim_2x`` maps the bundled 62.6-nm WF
input to the 31.3-nm SIM reference and is suitable for a local quantitative
screening, but its super-resolution task label is an experimental assumption.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_provenance import provenance, write_run_md


PROTOCOLS = {
    "upstream_deconv_1x": {
        "task_text": "deconvolution",
        "scale_factor": 1,
        "target_pixel_size": "62.6 x 62.6 nm",
        "scope": "Qualitative field-deletion check; the bundled SIM reference is 2x larger.",
        "source": "Upstream napari_fluoresfm.fluoresfm.test.predict example.",
    },
    "wf_to_sim_2x": {
        "task_text": "super-resolution with a scale factor of 2",
        "scale_factor": 2,
        "target_pixel_size": "31.3 x 31.3 nm",
        "scope": "Quantitative local screening against bundled SIM references; not the paper evaluation protocol.",
        "source": "Experimental mapping from 62.6-nm WF input to 31.3-nm SIM reference; task label is an explicit assumption.",
    },
}


def prompt(protocol: dict[str, object], *, task: bool = True, structure: bool = True, imaging: bool = True) -> str:
    """Build prompts by omitting exactly one semantic field when requested."""
    fields: list[str] = []
    if task:
        fields.append(f"Task: {protocol['task_text']}")
    fields.append("sample: fixed COS-7 cell line")
    if structure:
        fields.append("structure: microtubule")
    fields.append("fluorescence indicator: mEmerald (GFP)")
    if imaging:
        fields.extend(
            [
                "input microscope: wide-field microscope with excitation numerical aperture (NA) of 1.35, detection numerical aperture (NA) of 1.3",
                "input pixel size: 62.6 x 62.6 nm. Nearest interpolation with a factor of 2"
                if protocol["scale_factor"] == 2
                else "input pixel size: 62.6 x 62.6 nm",
                "target microscope: linear structured illumination microscope with excitation numerical aperture (NA) of 1.35, detection numerical aperture (NA) of 1.3",
                f"target pixel size: {protocol['target_pixel_size']}",
            ]
        )
    return "; ".join(fields) + "."


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=sorted(PROTOCOLS),
        default="wf_to_sim_2x",
        help="Use wf_to_sim_2x only as a documented local screening assumption.",
    )
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
    protocol = PROTOCOLS[args.protocol]
    plugin_source = root / "repos" / "napari-fluoresfm" / "src"
    if not plugin_source.is_dir():
        raise FileNotFoundError(f"Plugin source not found: {plugin_source}")
    sys.path.insert(0, str(plugin_source))
    from napari_fluoresfm.fluoresfm.test.predict import predict

    data_root = root / "example" / "data" / "BioSR_MT"
    source_index = data_root / "test.txt"
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
        "path_input": data_root / "test" / "channel_0" / "WF_noise_level_3",
        "path_input_index": input_index,
        "path_embedder": root / "example" / "checkpoints" / "biomedclip",
        "path_checkpoint": root / "example" / "checkpoints" / "fluoresfm" / "epoch_0_iter_700000.pt",
        "sf_lr": protocol["scale_factor"],
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "device": args.device,
        "compile": False,
    }
    for name, value in common.items():
        if name.startswith("path_") and not Path(value).exists():
            raise FileNotFoundError(f"Required asset missing ({name}): {value}")

    all_conditions = {
        "01_full": prompt(protocol),
        "02_no_task": prompt(protocol, task=False),
        "03_no_structure": prompt(protocol, structure=False),
        "04_no_imaging": prompt(protocol, imaging=False),
    }
    if args.protocol == "wf_to_sim_2x":
        canonical = (root / "example" / "data" / "text" / "train" / "dataset_text_ALL.txt").read_text(encoding="utf-8").splitlines()[0]
        if all_conditions["01_full"] != canonical:
            raise RuntimeError("The 2x BioSR_MT full prompt must match the bundled canonical training text exactly.")
    condition_keys = {
        "full": "01_full",
        "no_task": "02_no_task",
        "no_structure": "03_no_structure",
        "no_imaging": "04_no_imaging",
    }
    conditions = {
        condition_keys[name]: all_conditions[condition_keys[name]]
        for name in (args.conditions or condition_keys)
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runs_dir / "manifest.json"
    existing_conditions: dict[str, str] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_conditions = previous.get("conditions", {})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": args.protocol,
        "protocol_source": protocol["source"],
        "prompt_source": "example/data/text/train/dataset_text_ALL.txt for wf_to_sim_2x; field-deletion conditions omit exactly one field.",
        "parameters": {key: str(value) for key, value in common.items()},
        "conditions": {**existing_conditions, **conditions},
        "scope": protocol["scope"],
        "provenance": provenance(
            root=root,
            script=Path(__file__),
            assets={
                "input_index": input_index,
                "checkpoint": common["path_checkpoint"],
                "canonical_prompt": root / "example" / "data" / "text" / "train" / "dataset_text_ALL.txt",
            },
            parameters=common,
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_run_md(runs_dir / "run.md", manifest)

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
