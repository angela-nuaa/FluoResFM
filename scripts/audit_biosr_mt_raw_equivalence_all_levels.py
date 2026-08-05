#!/usr/bin/env python
"""Audit BioSR raw-MRC to bundled-example preprocessing equivalence."""

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


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace-root", default=Path.cwd(), type=Path)
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "biosr-mt-raw-equivalence/v1":
        raise ValueError("不支持的配置版本。")
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    raw_root, reference_root, output = (resolve(config["input"]["raw_root"]), resolve(config["reference_root"]), resolve(config["output"]))
    reader_path = resolve(config["input"]["mrc_reader"])
    if output.exists():
        raise FileExistsError(f"拒绝覆盖既有审计输出：{output}")
    spec = importlib.util.spec_from_file_location("biosr_mrc_reader", reader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法导入 MRC 读取器。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = tuple(config["acceptance"]["expected_shape_yx"])
    rows: list[dict[str, object]] = []
    for image_id in config["image_ids"]:
        for level in config["levels"]:
            source = raw_root / f"Cell_{int(image_id):03d}" / f"RawSIMData_level_{level:02d}.mrc"
            reference = reference_root / f"WF_noise_level_{level}" / f"{image_id}.tif"
            _, raw = module.read_mrc(str(source))
            reconstructed = raw.astype(np.float32).mean(axis=2, dtype=np.float32)
            observed = tifffile.imread(reference).squeeze().astype(np.float32)
            if reconstructed.shape != expected or observed.shape != expected:
                raise ValueError(f"{image_id}, level {level} 形状异常。")
            max_error = float(np.max(np.abs(reconstructed - observed)))
            mean_error = float(np.mean(np.abs(reconstructed - observed)))
            if max_error > float(config["acceptance"]["max_abs_error"]):
                raise ValueError(f"{image_id}, level {level} 未通过：最大绝对误差 {max_error}")
            rows.append({"image_id": image_id, "level": level, "raw_mrc_sha256": digest(source), "example_tiff_sha256": digest(reference), "max_abs_error": max_error, "mean_abs_error": mean_error})
        print(f"已完成样本 {image_id} 的 9 个等级像素审计。", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "pixel_equivalence.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    by_level = [{"level": level, "n_images": sum(r["level"] == level for r in rows), "max_abs_error": max(r["max_abs_error"] for r in rows if r["level"] == level)} for level in config["levels"]]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "范围": "BioSR_MT example 测试集 15 个样本、9 个信号等级的原始 MRC 至 example TIFF 像素等价审计。", "config": str(config_path.relative_to(root)), "config_sha256": digest(config_path), "script": str(Path(__file__).resolve().relative_to(root)), "script_sha256": digest(Path(__file__).resolve()), "n_comparisons": len(rows), "all_max_abs_error": max(r["max_abs_error"] for r in rows), "by_level": by_level}
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"全部完成：{len(rows)} / {len(rows)} 比较均完全一致。")


if __name__ == "__main__":
    main()
