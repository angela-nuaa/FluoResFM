#!/usr/bin/env python
"""Reconstruct and pixel-audit BioSR_MT example files from public raw MRC inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace-root", default=Path.cwd(), type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def load_mrc_reader(path: Path):
    spec = importlib.util.spec_from_file_location("biosr_mt_mrc_reader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 MRC 读取器：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_mrc


def main() -> None:
    args = parse_args()
    workspace = args.workspace_root.resolve()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "biosr-mt-raw-reconstruction/v1":
        raise ValueError("不支持的配置版本。")
    if config.get("status") != "candidate_not_frozen":
        raise ValueError("仅允许生成候选版本。")

    raw_root = resolve(workspace, config["input"]["raw_root"])
    reader_path = resolve(workspace, config["input"]["mrc_reader"])
    archive = resolve(workspace, config["input"]["archive"])
    output_root = resolve(workspace, config["output"]["root"])
    reference_input = resolve(workspace, config["reference"]["input_dir"])
    reference_target = resolve(workspace, config["reference"]["target_dir"])
    if not raw_root.is_dir() or not reader_path.is_file() or not archive.is_file():
        raise FileNotFoundError("缺少原始输入、MRC 读取器或已校验归档。")
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖既有输出：{output_root}")

    image_ids = config["reference"]["image_ids"]
    source_level = config["transform"]["source_level"]
    input_name = config["transform"]["input_source_name"]
    target_name = config["transform"]["target_source_name"]
    for image_id in image_ids:
        cell = raw_root / f"Cell_{int(image_id):03d}"
        required = (cell / input_name, cell / target_name, reference_input / f"{image_id}.tif", reference_target / f"{image_id}.tif")
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"{image_id} 缺少原始或参考文件。")

    preflight = {
        "样本数": len(image_ids),
        "原始等级": source_level,
        "变换": "RawSIM 第 3 轴 9 帧均值；SIM_gt 第 0 平面；不归一化；按读取器数组方向直接写入",
        "输出": str(output_root),
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    input_dir = output_root / "inputs" / f"WF_noise_level_{source_level}"
    target_dir = output_root / "targets" / "SIM"
    input_dir.mkdir(parents=True, exist_ok=False)
    target_dir.mkdir(parents=True, exist_ok=False)
    read_mrc = load_mrc_reader(reader_path)
    expected_input = tuple(config["acceptance"]["expected_input_shape_yx"])
    expected_target = tuple(config["acceptance"]["expected_target_shape_yx"])
    rows: list[dict[str, object]] = []

    for image_id in image_ids:
        cell = raw_root / f"Cell_{int(image_id):03d}"
        source_input = cell / input_name
        source_target = cell / target_name
        _, raw_xyz = read_mrc(str(source_input))
        _, target_xyz = read_mrc(str(source_target))
        raw = raw_xyz.astype(np.float32)
        target = target_xyz[:, :, int(config["transform"]["target_plane_index"])].astype(np.float32)
        reconstructed = raw.mean(axis=int(config["transform"]["input_frame_axis"]), dtype=np.float32)
        if reconstructed.shape != expected_input or target.shape != expected_target:
            raise ValueError(f"{image_id} 形状异常：输入 {reconstructed.shape}，目标 {target.shape}")
        output_input = input_dir / f"{image_id}.tif"
        output_target = target_dir / f"{image_id}.tif"
        tifffile.imwrite(output_input, reconstructed[None], metadata={"axes": "CYX"})
        tifffile.imwrite(output_target, target[None], metadata={"axes": "CYX"})
        reference_lr = tifffile.imread(reference_input / f"{image_id}.tif").squeeze().astype(np.float32)
        reference_hr = tifffile.imread(reference_target / f"{image_id}.tif").squeeze().astype(np.float32)
        lr_error = float(np.max(np.abs(reconstructed - reference_lr)))
        hr_error = float(np.max(np.abs(target - reference_hr)))
        if lr_error > float(config["acceptance"]["input_max_abs_error"]) or hr_error > float(config["acceptance"]["target_max_abs_error"]):
            raise ValueError(f"{image_id} 未通过像素等价校验：输入 {lr_error}，目标 {hr_error}")
        rows.append({
            "image_id": image_id,
            "cell": cell.name,
            "raw_input_sha256": sha256(source_input),
            "raw_target_sha256": sha256(source_target),
            "reconstructed_input_sha256": sha256(output_input),
            "reference_input_sha256": sha256(reference_input / f"{image_id}.tif"),
            "reconstructed_target_sha256": sha256(output_target),
            "reference_target_sha256": sha256(reference_target / f"{image_id}.tif"),
            "input_max_abs_error": lr_error,
            "target_max_abs_error": hr_error,
        })
        print(f"已完成 {image_id}：输入/目标像素等价校验通过。", flush=True)

    with (output_root / "pixel_equivalence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "BioSR_MT bundled example WF_noise_level_3 / SIM files reconstructed from verified public source files.",
        "config": str(config_path.relative_to(workspace)),
        "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve().relative_to(workspace)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "archive": str(archive.relative_to(workspace)),
        "archive_md5_expected": config["input"]["archive_md5"],
        "source_level": source_level,
        "n_images": len(rows),
        "all_input_max_abs_error": max(row["input_max_abs_error"] for row in rows),
        "all_target_max_abs_error": max(row["target_max_abs_error"] for row in rows),
        "python": sys.version,
        "numpy": np.__version__,
        "tifffile": tifffile.__version__,
    }
    with (output_root / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print("全部样本完成：候选复建数据已生成，像素等价审计通过。")


if __name__ == "__main__":
    main()
