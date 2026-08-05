"""Run semantically equivalent paraphrase experiment on BioSR_MT (P2-1).

Four conditions paraphrase the canonical ``wf_to_sim_2x`` prompt without
changing the physical meaning of any field.  The control (01_full) is the
canonical prompt verbatim.  Each subsequent condition rewords exactly one
semantic field (task / structure / imaging) while keeping all other text
identical; 05_full_paraphrase rewrites all three simultaneously.

Requires GPU for inference.  Evaluation, statistics, and plotting are done
locally in a separate step — this script only generates predictions.

This is the historical P2-1 implementation; consult its result document for the recorded protocol and boundary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_provenance import provenance, write_run_md


# ---------------------------------------------------------------------------
# Canonical text fragments (matching ``run_mt_field_ablation.py`` and
# ``example/data/text/train/dataset_text_ALL.txt`` line 1 for wf_to_sim_2x).
# ---------------------------------------------------------------------------

CANONICAL_TASK = "Task: super-resolution with a scale factor of 2"
CANONICAL_STRUCTURE = "structure: microtubule"
CANONICAL_IMAGING = [
    "input microscope: wide-field microscope with excitation numerical aperture (NA) of 1.35, "
    "detection numerical aperture (NA) of 1.3",
    "input pixel size: 62.6 x 62.6 nm. Nearest interpolation with a factor of 2",
    "target microscope: linear structured illumination microscope with excitation numerical aperture (NA) of 1.35, "
    "detection numerical aperture (NA) of 1.3",
    "target pixel size: 31.3 x 31.3 nm",
]
SHARED = [
    "sample: fixed COS-7 cell line",
    "fluorescence indicator: mEmerald (GFP)",
]


def build_prompt(
    task: str = CANONICAL_TASK,
    structure: str = CANONICAL_STRUCTURE,
    imaging: list[str] | None = None,
) -> str:
    """Reassemble the full prompt from individual semantic-field fragments."""
    if imaging is None:
        imaging = CANONICAL_IMAGING
    fields = [task, SHARED[0], structure, SHARED[1], *imaging]
    return "; ".join(fields) + "."


# ---------------------------------------------------------------------------
# Paraphrase definitions
#
# Each paraphrase preserves the exact physical meaning of the original field:
#   - Task keeps 2× WF→SIM super-resolution.
#   - Structure keeps microtubule / MT cytoskeleton.
#   - Imaging keeps the named linear structured-illumination target modality,
#     NA_ex=1.35, NA_det=1.3, input 62.6 nm, 2× nearest interpolation, and
#     output 31.3 nm.  It must not substitute a different SIM designation.
# ---------------------------------------------------------------------------

PARAPHRASE_TASK = "Task: 2× super-resolution from wide-field to SIM"
PARAPHRASE_STRUCTURE = "structure: microtubules (MT cytoskeleton)"
PARAPHRASE_IMAGING = [
    "illumination: wide-field (ex NA 1.35, det NA 1.3)",
    "acquisition: 62.6 nm/pixel with 2× nearest interpolation for matching",
    "target microscope: linear structured-illumination microscope (excitation NA 1.35, detection NA 1.3)",
    "output pixel size: 31.3 nm/pixel",
]


# Each entry: text, fields_changed, and a note explaining semantic equivalence.
CONDITIONS = {
    "01_full": {
        "text": build_prompt(),
        "fields_changed": [],
        "notes": "Canonical control — verbatim from dataset_text_ALL.txt line 1.",
    },
    "02_task_paraphrase": {
        "text": build_prompt(task=PARAPHRASE_TASK),
        "fields_changed": ["task"],
        "notes": (
            "Original 'super-resolution with a scale factor of 2' → "
            "'2× super-resolution from wide-field to SIM'.  "
            "Scale factor and direction (WF→SIM) are preserved."
        ),
    },
    "03_structure_paraphrase": {
        "text": build_prompt(structure=PARAPHRASE_STRUCTURE),
        "fields_changed": ["structure"],
        "notes": (
            "Original 'microtubule' → 'microtubules (MT cytoskeleton)'.  "
            "Same biological structure; singular→plural and explicit abbreviation."
        ),
    },
    "04_imaging_paraphrase": {
        "text": build_prompt(imaging=PARAPHRASE_IMAGING),
        "fields_changed": ["imaging"],
        "notes": (
            "All imaging parameters preserved: NA_ex=1.35, NA_det=1.3, "
            "input 62.6 nm/pixel, 2× nearest interpolation, the named linear "
            "structured-illumination target modality, and output 31.3 nm/pixel.  "
            "Only wording and field labels are rewritten; no alternate SIM modality is introduced."
        ),
    },
    "05_full_paraphrase": {
        "text": build_prompt(PARAPHRASE_TASK, PARAPHRASE_STRUCTURE, PARAPHRASE_IMAGING),
        "fields_changed": ["task", "structure", "imaging"],
        "notes": (
            "All three fields paraphrased simultaneously; combined form of "
            "the above individual paraphrases.  No physical parameters changed."
        ),
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only on the first 2 images to verify the pipeline.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    root, runs_dir = args.repo_root.resolve(), args.runs_dir.resolve()

    # --- Import inference engine -------------------------------------------
    plugin_source = root / "repos" / "napari-fluoresfm" / "src"
    if not plugin_source.is_dir():
        raise FileNotFoundError(f"Plugin source not found: {plugin_source}")
    sys.path.insert(0, str(plugin_source))
    from napari_fluoresfm.fluoresfm.test.predict import predict

    # --- Validate canonical prompt -----------------------------------------
    canonical_path = (
        root / "example" / "data" / "text" / "train" / "dataset_text_ALL.txt"
    )
    canonical_line = canonical_path.read_text(encoding="utf-8").splitlines()[0]
    expected_full = build_prompt()
    if expected_full != canonical_line:
        raise RuntimeError(
            f"01_full does not match {canonical_path} line 1.\n"
            f"  BUILT:  {expected_full}\n"
            f"  CANON:  {canonical_line}"
        )

    # --- Image index -------------------------------------------------------
    data_root = root / "example" / "data" / "BioSR_MT"
    source_index = data_root / "test.txt"
    all_images = source_index.read_text(encoding="utf-8").splitlines()

    if args.smoke_test:
        images = all_images[:2]
        print(
            f"SMOKE TEST: using {len(images)} of {len(all_images)} images: {images}"
        )
    else:
        images = all_images
        print(f"Full run: {len(images)} images from {source_index}")

    runs_dir.mkdir(parents=True, exist_ok=True)
    input_index = runs_dir / "input_index.txt"
    input_index.write_text("\n".join(images) + "\n", encoding="utf-8")

    common = {
        "path_input": data_root / "test" / "channel_0" / "WF_noise_level_3",
        "path_input_index": input_index,
        "path_embedder": root / "example" / "checkpoints" / "biomedclip",
        "path_checkpoint": (
            root
            / "example"
            / "checkpoints"
            / "fluoresfm"
            / "epoch_0_iter_700000.pt"
        ),
        "sf_lr": 2,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "device": args.device,
        "compile": False,
    }
    for name, value in common.items():
        if name.startswith("path_") and not Path(value).exists():
            raise FileNotFoundError(f"Required asset missing ({name}): {value}")

    # --- Manifest ----------------------------------------------------------
    paraphrase_records: dict[str, dict[str, object]] = {}
    for cond_name, cond in CONDITIONS.items():
        paraphrase_records[cond_name] = {
            "original_text": expected_full,
            "paraphrased_text": cond["text"],
            "fields_changed": cond["fields_changed"],
            "semantic_equivalence_notes": cond["notes"],
        }

    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "P2-1 semantic-equivalent paraphrase robustness (BioSR_MT)",
        "hypothesis": (
            "Semantically equivalent rewording of task / structure / imaging "
            "fields causes smaller output degradation relative to the canonical "
            "prompt than full field deletion observed in prior experiments."
        ),
        "canonical_source": str(canonical_path),
        "paraphrase_records": paraphrase_records,
        "parameters": {key: str(value) for key, value in common.items()},
        "conditions": {name: cond["text"] for name, cond in CONDITIONS.items()},
        "scope": (
            f"{'Smoke test (2 images)' if args.smoke_test else 'All 15 images'} "
            f"× {len(CONDITIONS)} conditions.  "
            "GPU inference only; evaluation is a separate local step."
        ),
        "provenance": provenance(
            root=root,
            script=Path(__file__),
            assets={
                "input_index": input_index,
                "checkpoint": common["path_checkpoint"],
                "canonical_prompt": canonical_path,
            },
            parameters=common,
        ),
    }

    manifest_path = runs_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_run_md(runs_dir / "run.md", manifest)

    # --- Run inference per condition ---------------------------------------
    for cond_name, cond in CONDITIONS.items():
        output = runs_dir / cond_name
        if output.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite {output}; use --overwrite."
                )
            shutil.rmtree(output)
        output.mkdir(parents=True)

        params: dict[str, object] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in common.items()
        }
        params.update({"path_output": str(output), "text": cond["text"]})

        changed = cond["fields_changed"]
        print(
            f"\n=== {cond_name} ===  fields: {changed if changed else 'none (control)'}"
        )
        print(f"{cond['text']}\n")

        if predict(params) != 1:
            raise RuntimeError(f"Prediction failed for {cond_name}")

    n_img = len(images)
    n_cond = len(CONDITIONS)
    print(
        f"\nDone.  {n_img} images × {n_cond} conditions "
        f"= {n_img * n_cond} predictions written to {runs_dir}"
    )


if __name__ == "__main__":
    main()
