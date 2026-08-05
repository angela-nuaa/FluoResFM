#!/usr/bin/env python
"""Audit all bundled BioSR_MT training patches against raw-MRC regeneration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(image: np.ndarray, low: float, high: float) -> np.ndarray:
    image = np.clip(image.astype(np.float32), 0.0, None)
    lower, upper = np.percentile(image, (low, high))
    scale = upper - lower
    if upper == 0:
        return image * 0.0
    if scale == 0:
        scale = 1.0
    return (image - lower) / scale


def check_grid(image_id: str, level: int, lr: np.ndarray, hr: np.ndarray, root: Path, grid: dict[str, int], tolerance: float) -> tuple[int, float]:
    lp, ls, hp, hs = (grid["lr_patch"], grid["lr_stride"], grid["hr_patch"], grid["hr_stride"])
    lr_dir = root / f"WF_noise_level_{level}_p{lp}_s{ls}_2d"
    hr_dir = root / f"SIM_p{hp}_s{hs}_2d"
    max_error, count = 0.0, 0
    for row, y in enumerate(range(0, lr.shape[0] - lp + 1, ls)):
        for col, x in enumerate(range(0, lr.shape[1] - lp + 1, ls)):
            name = f"{image_id}_{row}_{col}.tif"
            expected_lr = lr[y : y + lp, x : x + lp]
            expected_hr = hr[row * hs : row * hs + hp, col * hs : col * hs + hp]
            actual_lr = tifffile.imread(lr_dir / name).squeeze().astype(np.float32)
            actual_hr = tifffile.imread(hr_dir / name).squeeze().astype(np.float32)
            error = max(float(np.max(np.abs(expected_lr - actual_lr))), float(np.max(np.abs(expected_hr - actual_hr))))
            if error > tolerance:
                raise ValueError(f"{name}, level {level}, p{lp}: 误差 {error} 超过阈值 {tolerance}")
            max_error = max(max_error, error)
            count += 2
    return count, max_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace-root", default=Path.cwd(), type=Path)
    parser.add_argument("--image-ids", nargs="+", help="仅审计指定训练图像编号，用于可并行的 I/O 分片。")
    parser.add_argument("--output-suffix", default="", help="附加到配置输出目录的分片名称。")
    args = parser.parse_args()
    root, config_path = args.workspace_root.resolve(), args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "biosr-mt-train-patch-equivalence/v1":
        raise ValueError("不支持的配置版本。")
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    raw_root, reference_root, output = resolve(config["input"]["raw_root"]), resolve(config["reference_root"]), resolve(config["output"])
    if args.output_suffix:
        output = output / args.output_suffix
    if output.exists():
        raise FileExistsError(f"拒绝覆盖既有审计输出：{output}")
    ids = [Path(line).stem for line in resolve(config["train_index"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.image_ids:
        unknown = sorted(set(args.image_ids) - set(ids))
        if unknown:
            raise ValueError(f"指定图像不在 train.txt：{unknown}")
        ids = args.image_ids
    spec = importlib.util.spec_from_file_location("biosr_mrc_reader", resolve(config["input"]["mrc_reader"]))
    if spec is None or spec.loader is None:
        raise RuntimeError("无法导入 MRC 读取器。")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    norm = config["normalization"]; tolerance = float(config["acceptance"]["max_abs_error"])
    rows: list[dict[str, object]] = []
    for image_id in ids:
        cell = raw_root / f"Cell_{int(image_id):03d}"
        _, gt = module.read_mrc(str(cell / "SIM_gt.mrc"))
        hr = normalize(gt[:, :, 0], float(norm["low_percentile"]), float(norm["high_percentile"]))
        for level in config["levels"]:
            _, raw = module.read_mrc(str(cell / f"RawSIMData_level_{level:02d}.mrc"))
            lr = normalize(raw.mean(axis=2, dtype=np.float32), float(norm["low_percentile"]), float(norm["high_percentile"]))
            for grid in config["grids"]:
                checked, max_error = check_grid(image_id, level, lr, hr, reference_root, grid, tolerance)
                rows.append({"image_id": image_id, "level": level, "lr_patch": grid["lr_patch"], "hr_patch": grid["hr_patch"], "arrays_checked": checked, "max_abs_error": max_error})
        print(f"已完成训练样本 {image_id} 的全部 level 与 patch 网格审计。", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "patch_equivalence.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "范围": "BioSR_MT train.txt 的全部 30 个样本、9 个信号等级、两种 LR/HR patch 网格的原始 MRC 至 example patch 像素审计。", "config": str(config_path.relative_to(root)), "config_sha256": sha256(config_path), "script": str(Path(__file__).resolve().relative_to(root)), "script_sha256": sha256(Path(__file__).resolve()), "n_grid_conditions": len(rows), "arrays_checked": sum(int(row["arrays_checked"]) for row in rows), "all_max_abs_error": max(float(row["max_abs_error"]) for row in rows), "acceptance_tolerance": tolerance}
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"全部完成：{len(rows)} 个样本-level-grid 条件，{manifest['arrays_checked']} 个 patch 数组均通过。")


if __name__ == "__main__":
    main()
