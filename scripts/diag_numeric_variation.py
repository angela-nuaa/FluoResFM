#!/usr/bin/env python
"""Trace cross-process numeric variation in FluoResFM inference (diagnostic).

Runs the production inference path (napari predict) on WF_noise_level_3/41.tif
in a FRESH process and saves the candidate. Run several times (separate
processes) and compare outputs to determine whether the ~1e-3 run-to-run
difference between the batch run and deconv15 is random nondeterminism or a
state artifact.

Usage (on cloud, needs HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1):
  python scripts/diag_numeric_variation.py --output /tmp/diag_run1
  python scripts/diag_numeric_variation.py --output /tmp/diag_run2 ...
  python scripts/diag_numeric_variation.py --output /tmp/diag_det --deterministic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--double", action="store_true", help="run predict() twice in one process and compare (within-process determinism)")
    ap.add_argument("--deterministic", action="store_true", help="call torch.use_deterministic_algorithms(True) before predict")
    ap.add_argument("--benchmark-off", action="store_true", help="inject a runtime-patched predict module with cudnn.benchmark=False (never touches the source file)")
    ap.add_argument("--workspace-root", default=Path.cwd())
    args = ap.parse_args()

    root = Path(args.workspace_root)
    sys.path.insert(0, str(root / "repos/napari-fluoresfm/src"))

    deterministic_note = None
    if args.deterministic:
        try:
            torch.use_deterministic_algorithms(True)
            deterministic_note = "use_deterministic_algorithms=True set"
        except Exception as exc:  # noqa: BLE001
            deterministic_note = f"deterministic setup error: {exc}"

    if args.benchmark_off:
        # 运行时注入：把 predict.py 源码的 `cudnn.benchmark = True` 替换为 False，
        # exec 编译后塞进 sys.modules。磁盘源码保持逐字节不变。
        predict_source = (root / "repos/napari-fluoresfm/src/napari_fluoresfm/fluoresfm/test/predict.py").read_text(encoding="utf-8")
        if "torch.backends.cudnn.benchmark = True" not in predict_source:
            raise RuntimeError("predict.py 源码结构变化，无法应用 benchmark-off 补丁。")
        patched_source = predict_source.replace(
            "torch.backends.cudnn.benchmark = True",
            "torch.backends.cudnn.benchmark = False",
        )
        import types
        ns: dict = {}
        exec(compile(patched_source, "predict_benchmark_off.py", "exec"), ns)
        patched_module = types.ModuleType("napari_fluoresfm.fluoresfm.test.predict")
        patched_module.__dict__.update(ns)
        sys.modules["napari_fluoresfm.fluoresfm.test.predict"] = patched_module

    from napari_fluoresfm.fluoresfm.test.predict import predict

    input_dir = root / "example/data/BioSR_MT/test/channel_0/WF_noise_level_3"
    out = Path(args.output)
    if out.exists():
        print(json.dumps({"error": f"output exists, refusing overwrite: {out}"}))
        return 2
    out.mkdir(parents=True)
    index = out / "index.txt"
    index.write_text("41.tif\n")

    params = {
        "path_input": str(input_dir),
        "path_input_index": str(index),
        "path_output": str(out),
        "path_embedder": str(root / "example/checkpoints/biomedclip"),
        "path_checkpoint": str(root / "example/checkpoints/fluoresfm/epoch_0_iter_700000.pt"),
        "sf_lr": 1,
        "batch_size": 8,
        "patch_size": 64,
        "device": args.device,
        "compile": False,
        "text": "Task: deconvolution; sample: fixed COS-7 cell line; structure: microtubule; fluorescence indicator: mEmerald (GFP); input microscope: wide-field microscope with excitation numerical aperture (NA) of 1.35, detection numerical aperture (NA) of 1.3; input pixel size: 62.6 x 62.6 nm; target microscope: linear structured illumination microscope with excitation numerical aperture (NA) of 1.35, detection numerical aperture (NA) of 1.3; target pixel size: 62.6 x 62.6 nm.",
    }
    try:
        result = predict(params)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": type(exc).__name__, "message": str(exc), "deterministic_note": deterministic_note}))
        return 1
    cand = out / "41.tif"
    sha = hashlib.sha256(cand.read_bytes()).hexdigest()
    a = tifffile.imread(cand)
    report = {
        "result": result,
        "output": str(cand),
        "sha256": sha,
        "sha16": sha[:16],
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "deterministic": args.deterministic,
        "deterministic_note": deterministic_note,
        "device": args.device,
    }
    if args.double:
        out2 = out / "second"
        out2.mkdir()
        index2 = out2 / "index.txt"
        index2.write_text("41.tif\n")
        params2 = dict(params)
        params2["path_input_index"] = str(index2)
        params2["path_output"] = str(out2)
        result2 = predict(params2)
        cand2 = out2 / "41.tif"
        a2 = tifffile.imread(cand2)
        d = np.abs(a - a2)
        report["within_process"] = {
            "result2": result2,
            "second_sha16": hashlib.sha256(cand2.read_bytes()).hexdigest()[:16],
            "byte_identical": sha == hashlib.sha256(cand2.read_bytes()).hexdigest(),
            "max_abs_error": float(d.max()),
            "mean_abs_error": float(d.mean()),
        }
    print(json.dumps(report))
    return 0 if result == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
