#!/usr/bin/env python
"""Run one non-overwriting upstream-inference probe for a BioSR_MT _fluoresfm TIFF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace-root", default=Path.cwd(), type=Path)
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    root, config_path = args.workspace_root.resolve(), args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "biosr-mt-fluoresfm-inference-probe/v1":
        raise ValueError("不支持的配置版本。")
    output_root = resolve(root, config["output"])
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖既有输出：{output_root}")
    input_root, reference_root = resolve(root, config["input_root"]), resolve(root, config["reference_root"])
    source_root, napari_repo = resolve(root, config["napari_source_root"]), resolve(root, config["napari_repository"])
    embedder_root, checkpoint = resolve(root, config["embedder_root"]), resolve(root, config["checkpoint"])
    required = [input_root, reference_root, source_root, napari_repo, embedder_root, checkpoint]
    if not all(path.exists() for path in required):
        raise FileNotFoundError("缺少输入：" + "; ".join(str(path) for path in required if not path.exists()))
    image_ids = [str(value) for value in config["image_ids"]]
    names = [f"{image_id}.tif" for image_id in image_ids]
    if not all((input_root / name).is_file() and (reference_root / name).is_file() for name in names):
        raise FileNotFoundError("输入或参考 TIFF 不完整。")

    output_root.mkdir(parents=True, exist_ok=False)
    candidate_root = output_root / "candidate"
    candidate_root.mkdir()
    index_path = output_root / "input_index.txt"
    index_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    sys.path.insert(0, str(source_root))
    from napari_fluoresfm.fluoresfm.test.predict import predict

    params = {
        "path_input": str(input_root),
        "path_input_index": str(index_path),
        "path_output": str(candidate_root),
        "path_embedder": str(embedder_root),
        "path_checkpoint": str(checkpoint),
        "sf_lr": int(config["scale_factor"]),
        "batch_size": int(config["batch_size"]),
        "patch_size": int(config["patch_size"]),
        "device": config["device"],
        "compile": bool(config["compile"]),
        "text": config["prompt"],
    }
    result = predict(params)
    if result != 1:
        raise RuntimeError(f"上游推理返回失败状态：{result}")
    rows = []
    for name in names:
        candidate, reference = candidate_root / name, reference_root / name
        if not candidate.is_file():
            raise FileNotFoundError(f"推理未生成：{candidate}")
        actual, expected = tifffile.imread(candidate), tifffile.imread(reference)
        compatible = actual.shape == expected.shape and actual.dtype == expected.dtype
        error = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))) if compatible else None
        rows.append({"file": name, "shape_match": actual.shape == expected.shape, "dtype_match": actual.dtype == expected.dtype, "max_abs_error": error, "mean_abs_error": float(np.mean(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))) if compatible else None, "candidate_sha256": sha256(candidate), "reference_sha256": sha256(reference)})
    with (output_root / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "单图上游 FluoResFM 推理探针；用于识别 bundled _fluoresfm 文件的生成链，不覆盖 bundled example。",
        "config": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256(config_path),
        "script": Path(__file__).resolve().relative_to(root).as_posix(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "napari_revision": git_revision(napari_repo),
        "checkpoint_sha256": sha256(checkpoint),
        "embedder_config_sha256": sha256(embedder_root / "open_clip_config.json"),
        "params": params,
        "comparison": rows,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
