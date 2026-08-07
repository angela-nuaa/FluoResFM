#!/usr/bin/env python
"""Freeze and safely reproduce the audited BioSR_MT bundled-example assets.

The source bundled directory is read-only.  This script creates a new derived
run directory, inventories every file in the source directory (including the
currently unresolved ``*_fluoresfm`` test derivatives), and materializes only
the full-frame, training-patch, and index assets with an already audited raw
MRC transform.  It refuses to overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile


SCHEMA_VERSION = "biosr-mt-verified-materialization/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace-root", default=Path.cwd(), type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the planned scope without writing output.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_mrc_reader(path: Path):
    spec = importlib.util.spec_from_file_location("biosr_mt_mrc_reader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 MRC 读取器：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_mrc


def normalized(image: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    image = np.clip(image.astype(np.float32), float(settings["negative_clip"]), None)
    lower, upper = np.percentile(image, (float(settings["low_percentile"]), float(settings["high_percentile"])))
    scale = upper - lower
    if upper == 0:
        return image * 0.0
    if scale == 0:
        scale = 1.0
    return (image - lower) / scale


def index_names(image_ids: Iterable[str], patch: int, stride: int) -> list[str]:
    grid = (502 - patch) // stride + 1
    return [f"{image_id}_{row}_{col}.tif" for image_id in image_ids for row in range(grid) for col in range(grid)]


def expected_indexes(test_ids: list[str], train_ids: list[str]) -> dict[Path, bytes]:
    def encode(names: list[str], *, final_newline: bool) -> bytes:
        suffix = "\r\n" if final_newline else ""
        return ("\r\n".join(names) + suffix).encode("utf-8")

    return {
        Path("test.txt"): encode([f"{image_id}.tif" for image_id in test_ids], final_newline=False),
        Path("train.txt"): encode([f"{image_id}.tif" for image_id in train_ids], final_newline=False),
        Path("test_p64_s64_2d.txt"): encode(index_names(test_ids, 64, 64), final_newline=True),
        Path("train_p32_s32_2d.txt"): encode(index_names(train_ids, 32, 32), final_newline=True),
        Path("train_p64_s64_2d.txt"): encode(index_names(train_ids, 64, 64), final_newline=True),
        Path("train_p128_s128_2d.txt"): encode(index_names(train_ids, 64, 64), final_newline=True),
    }


def inventory_reference(reference_root: Path, output_root: Path) -> dict[str, int]:
    """Write a complete, read-only baseline inventory of the bundled directory."""
    rows: list[dict[str, object]] = []
    tiff_count = text_count = other_count = 0
    for path in sorted(candidate for candidate in reference_root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(reference_root).as_posix()
        row: dict[str, object] = {
            "relative_path": relative,
            "kind": "other",
            "byte_size": path.stat().st_size,
            "file_sha256": sha256_file(path),
            "dtype": "",
            "shape": "",
            "array_sha256": "",
        }
        if path.suffix.lower() in {".tif", ".tiff"}:
            array = tifffile.imread(path)
            row.update({"kind": "tiff", "dtype": str(array.dtype), "shape": json.dumps(list(array.shape)), "array_sha256": array_sha256(array)})
            tiff_count += 1
        elif path.suffix.lower() == ".txt":
            row["kind"] = "text"
            text_count += 1
        else:
            other_count += 1
        rows.append(row)

    baseline_dir = output_root / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=False)
    with (baseline_dir / "reference_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"files": len(rows), "tiff_count": tiff_count, "text_count": text_count, "other_count": other_count}


def inventory_raw_sources(config: dict[str, Any], root: Path, output_root: Path) -> dict[str, int]:
    raw = config["input"]
    test_root = resolve(root, raw["test_raw_root"])
    train_root = resolve(root, raw["train_raw_root"])
    rows: list[dict[str, object]] = []
    for split, image_ids, raw_root in (("test", config["test_image_ids"], test_root), ("train", config["train_full_image_ids"], train_root)):
        for image_id in image_ids:
            cell = raw_root / f"Cell_{int(image_id):03d}"
            for level in config["levels"]:
                source = cell / f"RawSIMData_level_{level:02d}.mrc"
                rows.append({"split": split, "image_id": image_id, "kind": "rawsim", "level": level, "relative_path": source.relative_to(root).as_posix(), "byte_size": source.stat().st_size, "sha256": sha256_file(source)})
            target = cell / "SIM_gt.mrc"
            rows.append({"split": split, "image_id": image_id, "kind": "sim_gt", "level": "", "relative_path": target.relative_to(root).as_posix(), "byte_size": target.stat().st_size, "sha256": sha256_file(target)})
    with (output_root / "baseline" / "raw_source_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"files": len(rows)}


def compare_arrays(actual_path: Path, reference_path: Path, tolerance: float) -> float:
    actual = tifffile.imread(actual_path)
    reference = tifffile.imread(reference_path)
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError(f"数组形状或 dtype 不一致：{actual_path} 对比 {reference_path}")
    error = float(np.max(np.abs(actual.astype(np.float32) - reference.astype(np.float32))))
    if error > tolerance:
        raise ValueError(f"像素误差超过阈值 {tolerance}：{actual_path}，最大绝对误差 {error}")
    return error


def materialize(config: dict[str, Any], root: Path, output_root: Path) -> dict[str, object]:
    reference_root = resolve(root, config["reference_root"])
    raw = config["input"]
    read_mrc = load_mrc_reader(resolve(root, raw["mrc_reader"]))
    materialized_root = output_root / "materialized" / "BioSR_MT"
    rows: list[dict[str, object]] = []
    tiff_count = text_count = 0
    full_tolerance = float(config["acceptance"]["full_frame_max_abs_error"])
    patch_tolerance = float(config["acceptance"]["patch_max_abs_error"])

    def reference_storage_spec(relative: Path) -> tuple[np.dtype, bool]:
        reference = tifffile.imread(reference_root / relative)
        if reference.ndim == 2:
            return reference.dtype, False
        if reference.ndim == 3 and reference.shape[0] == 1:
            return reference.dtype, True
        raise ValueError(f"不支持的参考 TIFF 存储形状：{relative}，{reference.shape}")

    def write(array: np.ndarray, relative: Path, tolerance: float) -> None:
        nonlocal tiff_count
        destination, reference = materialized_root / relative, reference_root / relative
        dtype, singleton_channel = reference_storage_spec(relative)
        stored = array.astype(dtype, copy=False)
        if singleton_channel:
            stored = stored[None]
        destination.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(destination, stored, metadata={"axes": "CYX"} if singleton_channel else None)
        error = compare_arrays(destination, reference, tolerance)
        rows.append({"relative_path": relative.as_posix(), "kind": "tiff", "tolerance": tolerance, "max_abs_error": error, "output_file_sha256": sha256_file(destination), "reference_file_sha256": sha256_file(reference)})
        tiff_count += 1

    for split, image_ids, raw_root_string in (("test", config["test_image_ids"], raw["test_raw_root"]), ("train", config["train_full_image_ids"], raw["train_raw_root"])):
        raw_root = resolve(root, raw_root_string)
        for image_id in image_ids:
            cell = raw_root / f"Cell_{int(image_id):03d}"
            _, sim = read_mrc(str(cell / "SIM_gt.mrc"))
            sim_relative = Path(split) / "channel_0" / "SIM" / f"{image_id}.tif"
            sim_dtype, _ = reference_storage_spec(sim_relative)
            write(sim[:, :, 0].astype(sim_dtype, copy=False), sim_relative, full_tolerance)
            for level in config["levels"]:
                _, frames = read_mrc(str(cell / f"RawSIMData_level_{level:02d}.mrc"))
                wf_relative = Path(split) / "channel_0" / f"WF_noise_level_{level}" / f"{image_id}.tif"
                wf_dtype, _ = reference_storage_spec(wf_relative)
                wf = frames.astype(wf_dtype, copy=False).mean(axis=2, dtype=wf_dtype)
                write(wf, wf_relative, full_tolerance)

    train_ids = [Path(line).stem for line in resolve(root, config["train_index"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    if train_ids != [str(value) for value in range(1, 31)]:
        raise ValueError("当前已审计的训练 patch 范围必须严格对应 train.txt 中的样本 1–30。")
    train_raw_root = resolve(root, raw["train_raw_root"])
    for image_id in train_ids:
        cell = train_raw_root / f"Cell_{int(image_id):03d}"
        _, sim = read_mrc(str(cell / "SIM_gt.mrc"))
        hr = normalized(sim[:, :, 0], config["normalization"])
        cached_lr: dict[int, np.ndarray] = {}
        for level in config["levels"]:
            _, frames = read_mrc(str(cell / f"RawSIMData_level_{level:02d}.mrc"))
            cached_lr[level] = normalized(frames.mean(axis=2, dtype=np.float32), config["normalization"])
        for grid in config["patch_grids"]:
            lp, ls, hp, hs = (int(grid["lr_patch"]), int(grid["lr_stride"]), int(grid["hr_patch"]), int(grid["hr_stride"]))
            for row, y in enumerate(range(0, 502 - lp + 1, ls)):
                for col, x in enumerate(range(0, 502 - lp + 1, ls)):
                    name = f"{image_id}_{row}_{col}.tif"
                    for level, lr in cached_lr.items():
                        write(lr[y : y + lp, x : x + lp], Path("train") / "channel_0" / f"WF_noise_level_{level}_p{lp}_s{ls}_2d" / name, patch_tolerance)
                    write(hr[row * hs : row * hs + hp, col * hs : col * hs + hp], Path("train") / "channel_0" / f"SIM_p{hp}_s{hs}_2d" / name, patch_tolerance)

    for relative, content in expected_indexes(config["test_image_ids"], train_ids).items():
        destination, reference = materialized_root / relative, reference_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        if destination.read_bytes() != reference.read_bytes():
            raise ValueError(f"索引内容或编码不一致：{relative}")
        rows.append({"relative_path": relative.as_posix(), "kind": "text", "tolerance": "", "max_abs_error": "", "output_file_sha256": sha256_file(destination), "reference_file_sha256": sha256_file(reference)})
        text_count += 1

    text_metadata = config["text_metadata"]
    metadata_files = {Path("test") / "channel_0" / "SIM" / "text.txt": text_metadata["sim_text"]}
    metadata_files.update({Path("test") / "channel_0" / f"WF_noise_level_{level}" / "text.txt": text_metadata["wf_text"] for level in config["levels"]})
    for relative, content in metadata_files.items():
        destination, reference = materialized_root / relative, reference_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content.encode("utf-8"))
        if destination.read_bytes() != reference.read_bytes():
            raise ValueError(f"元数据文本内容或编码不一致：{relative}")
        rows.append({"relative_path": relative.as_posix(), "kind": "text", "tolerance": "", "max_abs_error": "", "output_file_sha256": sha256_file(destination), "reference_file_sha256": sha256_file(reference)})
        text_count += 1

    with (output_root / "materialization_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"materialized_root": materialized_root.relative_to(root).as_posix(), "tiff_count": tiff_count, "text_count": text_count, "max_abs_error": max(float(row["max_abs_error"]) for row in rows if row["kind"] == "tiff")}


def main() -> None:
    args = parse_args()
    root, config_path = args.workspace_root.resolve(), args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("不支持的配置版本。")
    output_root = resolve(root, config["output"])
    reference_root = resolve(root, config["reference_root"])
    required = [reference_root, resolve(root, config["input"]["test_raw_root"]), resolve(root, config["input"]["train_raw_root"]), resolve(root, config["input"]["mrc_reader"]), resolve(root, config["input"]["archive"])]
    if not all(path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError("缺少所需输入：" + "; ".join(missing))
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖既有输出：{output_root}")
    if args.dry_run:
        print(json.dumps({"output": str(output_root), "reference_root": str(reference_root), "scope": "完整基线 + 已审计的全图、训练 patch 与索引", "overwrite": "refused"}, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=False)
    baseline = inventory_reference(reference_root, output_root)
    expected = config["acceptance"]
    if baseline["tiff_count"] != int(expected["expected_reference_tiff_count"]) or baseline["text_count"] != int(expected["expected_reference_text_count"]):
        raise ValueError(f"参考目录计数意外变化：{baseline}")
    raw_sources = inventory_raw_sources(config, root, output_root)
    archive = resolve(root, config["input"]["archive"])
    materialized = materialize(config, root, output_root)
    if materialized["tiff_count"] != int(expected["expected_materialized_tiff_count"]) or materialized["text_count"] != int(expected["expected_materialized_text_count"]):
        raise ValueError(f"已生产文件计数错误：{materialized}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "阶段 1–2：完整 bundled reference 基线与已验证 BioSR_MT 资产的无覆盖生产。未生成 *_fluoresfm 测试派生图及其测试 patch。",
        "config": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "script": Path(__file__).resolve().relative_to(root).as_posix(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "archive": archive.relative_to(root).as_posix(),
        "archive_md5_expected": config["input"]["archive_md5"],
        "archive_sha256": sha256_file(archive),
        "reference_baseline": baseline,
        "raw_source_baseline": raw_sources,
        "materialized": materialized,
        "runtime": {"python": sys.version.replace("\n", " "), "platform": platform.platform(), "numpy": np.__version__, "tifffile": tifffile.__version__},
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "run.md").write_text(
        "# BioSR_MT 已验证资产生产记录\n\n"
        "本运行仅在新的 `data/derived/` 目录中写入文件，未修改 bundled example、原始 MRC 或既有审计证据。\n\n"
        "- 阶段 1：`baseline/reference_inventory.csv` 冻结了整个 bundled 目录（包括尚未严格生产的 `_fluoresfm` 测试派生资产）。\n"
        "- 阶段 2：`materialized/BioSR_MT/` 仅包含已验证的测试/训练全图、训练 patch 和索引。\n"
        "- `materialization_validation.csv` 是逐文件比较结果；`manifest.json` 为机器可读总记录。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["materialized"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
