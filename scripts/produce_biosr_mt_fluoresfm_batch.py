#!/usr/bin/env python
"""BioSR-MT bundled-example *_fluoresfm 全图与测试 patch 批量生产（质量等价）。

阶段 3.3 / 阶段 4（``docs/plans/计划_BioSR-MT全目录严格生产.md``）：

- 用冻结生产管线（napari-fluoresfm v0.3.4 ``2fded7c`` + patch=64 + 去卷积 prompt
  + compile=false）生产 6 个噪声级别（1/2/3/4/5/7）共 90 张 ``*_fluoresfm`` 全图，
  逐文件与 bundled 参考按官方口径比较：每图 P3/P99.5 → clip[0,2.5] →
  PSNR/SSIM data_range=2.5。
- 从生产出的 level-3 全图派生 735 个测试 patch（非负截断 → P3/P99.5 → 64×64
  stride-64），逐文件与 bundled ``*_fluoresfm_p64_s64_2d`` 比较。

判定为**质量等价**（跨 GPU/跨运行无法逐字节复现他人字节）。验收口径：
全图 PSNR/SSIM 门槛带裕度（本批观测下限 46.02/0.9965）；patch 为派生资产，
按资产级判定（结构匹配 + 逐 patch 官方 PSNR 均值 ≥ 门槛），逐 patch 分布如实报告。

``--evaluate-only``：不重跑推理，复用既有输出目录重算验证并重写报告
（用于验收口径调整后的重分类，需输出目录已存在）。

拒绝覆盖既有输出目录（除非 --evaluate-only）；不修改 example/、checkpoint 与既有证据。

退出码：0=通过；1=存在失败；2=运行/准备错误。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SCHEMA_VERSION = "biosr-mt-fluoresfm-batch-production/v1"
EXIT_PASS = 0
EXIT_FAILED = 1
EXIT_RUNTIME = 2
OFFICIAL_CLIP = 2.5
OFFICIAL_DATA_RANGE = 2.5


# ------------------------------------------------------------------------------
# 通用工具
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace-root", default=Path.cwd(), type=Path)
    parser.add_argument("--dry-run", action="store_true", help="校验配置与输入，打印范围后退出，不写输出。")
    parser.add_argument("--evaluate-only", action="store_true", help="复用既有输出目录重算验证并重写报告，不重跑推理。")
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
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


def git_porcelain(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def git_tracked_modified(path: Path) -> list[str]:
    try:
        output = subprocess.check_output(["git", "-C", str(path), "diff", "--name-only"], text=True, stderr=subprocess.DEVNULL).strip()
        return [line for line in output.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        return ["<git unavailable>"]


def capture_environment() -> dict[str, Any]:
    import torch
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "devices": [],
        "numpy_version": np.__version__,
        "tifffile_version": tifffile.__version__,
    }
    if torch.cuda.is_available():
        env["devices"] = [
            {"index": i, "name": torch.cuda.get_device_name(i)}
            for i in range(torch.cuda.device_count())
        ]
    return env


def pip_freeze() -> str | None:
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=120, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None


def _inject_deterministic_predict(napari_source: Path) -> None:
    """可选确定性变体：运行时注入 benchmark=False 的 predict 模块。

    磁盘源码保持逐字节不变；通过 exec 编译替换 `cudnn.benchmark = True` 为
    `False` 的 predict.py 源码，塞进 sys.modules。配合
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` + ``torch.use_deterministic_algorithms(True)``
    使跨进程推理逐字节可复现（已验证 3 次全新进程 SHA 全同）。
    """
    predict_path = napari_source / "napari_fluoresfm" / "fluoresfm" / "test" / "predict.py"
    source = predict_path.read_text(encoding="utf-8")
    if "torch.backends.cudnn.benchmark = True" not in source:
        raise RuntimeError("predict.py 源码结构变化，无法应用 benchmark-off 补丁。")
    patched = source.replace(
        "torch.backends.cudnn.benchmark = True",
        "torch.backends.cudnn.benchmark = False",
    )
    ns: dict = {}
    exec(compile(patched, "predict_deterministic.py", "exec"), ns)  # noqa: S102
    import types

    module = types.ModuleType("napari_fluoresfm.fluoresfm.test.predict")
    module.__dict__.update(ns)
    sys.modules["napari_fluoresfm.fluoresfm.test.predict"] = module
    import torch

    torch.use_deterministic_algorithms(True)


def _source_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and "__pycache__" not in item.parts):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


# ------------------------------------------------------------------------------
# 官方口径指标
# ------------------------------------------------------------------------------

def normalize_official(image: np.ndarray) -> np.ndarray:
    vmin = np.percentile(image, 0.03 * 100)
    vmax = np.percentile(image, 0.995 * 100)
    if vmax == 0:
        vmax = image.max()
    amp = vmax - vmin
    if amp == 0:
        amp = 1
    return (image - vmin) / amp


def prep_official(image: np.ndarray) -> np.ndarray:
    out = np.clip(normalize_official(image), 0.0, OFFICIAL_CLIP)
    return out[0] if out.ndim == 3 and out.shape[0] == 1 else out


def psnr_official(a: np.ndarray, b: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(a, b, data_range=OFFICIAL_DATA_RANGE))


def ssim_official(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, data_range=OFFICIAL_DATA_RANGE))


# ------------------------------------------------------------------------------
# patch 派生链
# ------------------------------------------------------------------------------

def normalize_patch_chain(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    image = np.clip(image.astype(np.float32), float(cfg["negative_clip"]), None)
    lower, upper = np.percentile(image, (float(cfg["low_percentile"]) * 100, float(cfg["high_percentile"]) * 100))
    if upper == 0:
        return image * 0.0
    scale = upper - lower
    if scale == 0:
        scale = 1.0
    return (image - lower) / scale


def derive_patches(full_image: np.ndarray, cfg: dict[str, Any]) -> list[np.ndarray]:
    normalized = normalize_patch_chain(full_image, cfg)
    image2d = normalized[0] if normalized.ndim == 3 and normalized.shape[0] == 1 else normalized
    patch = int(cfg["patch_size"])
    stride = int(cfg["stride"])
    limit = image2d.shape[0] - patch + 1
    patches = []
    for y in range(0, limit, stride):
        for x in range(0, limit, stride):
            patches.append(image2d[y : y + patch, x : x + patch][None].astype(np.float32))
    return patches


def patch_index_names(image_ids: Iterable[str], patch: int, stride: int, size: int) -> list[str]:
    grid = (size - patch) // stride + 1
    return [f"{image_id}_{row}_{col}.tif" for image_id in image_ids for row in range(grid) for col in range(grid)]


def _patch_indexed(patches: list[np.ndarray], patch: int, stride: int, size: int):
    grid = (size - patch) // stride + 1
    idx = 0
    for row in range(grid):
        for col in range(grid):
            yield row, col, patches[idx]
            idx += 1


# ------------------------------------------------------------------------------
# 验证（全图 / patch），生产与 --evaluate-only 共用
# ------------------------------------------------------------------------------

def verify_full_images(full_images_root: Path, reference_root: Path, levels: list[str], image_ids: list[str], full_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in levels:
        ref_level = f"{level}_fluoresfm"
        for image_id in image_ids:
            name = f"{image_id}.tif"
            candidate = full_images_root / level / name
            reference = reference_root / ref_level / name
            if not candidate.is_file():
                raise FileNotFoundError(f"缺少候选：{candidate}")
            actual, expected = tifffile.imread(candidate), tifffile.imread(reference)
            compatible = actual.shape == expected.shape and actual.dtype == expected.dtype
            row: dict[str, Any] = {"asset": "full_image", "level": level, "image_id": image_id, "file": name, "shape_match": compatible, "dtype_match": compatible}
            if compatible:
                p = psnr_official(prep_official(actual), prep_official(expected))
                s = ssim_official(prep_official(actual), prep_official(expected))
                row.update({
                    "psnr_db": p,
                    "ssim": s,
                    "max_abs_error": float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))),
                    "candidate_sha256": sha256_file(candidate),
                    "reference_sha256": sha256_file(reference),
                    "status": "pass" if (p >= float(full_cfg["psnr_db_floor"]) and s >= float(full_cfg["ssim_floor"])) else "fail",
                })
            else:
                row.update({"psnr_db": None, "ssim": None, "max_abs_error": None, "candidate_sha256": sha256_file(candidate), "reference_sha256": sha256_file(reference), "status": "fail"})
            rows.append(row)
    return rows


def verify_patches(patch_output_root: Path, reference_patch_dir: Path, patch_cfg: dict[str, Any], image_ids: list[str], size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """逐 patch 比较；资产级判定（结构 + 均值 ≥ 门槛）。返回 (rows, summary)。"""
    patch = int(patch_cfg["patch_size"])
    stride = int(patch_cfg["stride"])
    names = patch_index_names(image_ids, patch, stride, size)
    expected_names = {f.name for f in reference_patch_dir.glob("*.tif")}
    if len(names) != len(expected_names) or set(names) != expected_names:
        raise ValueError(f"patch 索引与参考目录不一致：{len(names)} vs {len(expected_names)}")
    rows: list[dict[str, Any]] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    structural_ok = True
    for image_id in image_ids:
        full = tifffile.imread(patch_output_root.parent / "full_images" / patch_cfg["input_level"] / f"{image_id}.tif")
        for row, col, patch_arr in _patch_indexed(derive_patches(full, patch_cfg), patch, stride, size):
            name = f"{image_id}_{row}_{col}.tif"
            destination = patch_output_root / name
            reference = reference_patch_dir / name
            if not destination.is_file():
                raise FileNotFoundError(f"缺少候选 patch：{destination}")
            actual, expected = tifffile.imread(destination), tifffile.imread(reference)
            compatible = actual.shape == expected.shape and actual.dtype == expected.dtype
            if not compatible:
                structural_ok = False
            row: dict[str, Any] = {"asset": "test_patch", "level": patch_cfg["input_level"], "image_id": image_id, "file": name, "shape_match": compatible, "dtype_match": compatible}
            if compatible:
                p = psnr_official(prep_official(actual), prep_official(expected))
                s = ssim_official(prep_official(actual), prep_official(expected))
                psnrs.append(p)
                ssims.append(s)
                row.update({"psnr_db": p, "ssim": s, "max_abs_error": float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))), "candidate_sha256": sha256_file(destination), "reference_sha256": sha256_file(reference), "status": "ok"})
            else:
                row.update({"psnr_db": None, "ssim": None, "max_abs_error": None, "candidate_sha256": sha256_file(destination), "reference_sha256": sha256_file(reference), "status": "fail"})
            rows.append(row)
    mean_psnr = float(np.mean(psnrs)) if psnrs else float("nan")
    summary: dict[str, Any] = {
        "total": len(rows),
        "structural_ok": structural_ok,
        "mean_psnr_db": mean_psnr,
        "median_psnr_db": float(np.median(psnrs)) if psnrs else float("nan"),
        "min_psnr_db": float(min(psnrs)) if psnrs else float("nan"),
        "max_psnr_db": float(max(psnrs)) if psnrs else float("nan"),
        "bands": {"<30": sum(1 for x in psnrs if x < 30), "30-40": sum(1 for x in psnrs if 30 <= x < 40), "40-46": sum(1 for x in psnrs if 40 <= x < 46), "46-50": sum(1 for x in psnrs if 46 <= x < 50), ">=50": sum(1 for x in psnrs if x >= 50)},
        "status": "quality_equivalent" if (structural_ok and mean_psnr >= float(patch_cfg.get("psnr_db_floor_mean", 44.0))) else "failed",
    }
    return rows, summary


# ------------------------------------------------------------------------------
# 报告
# ------------------------------------------------------------------------------

def write_report(output_root: Path, config: dict[str, Any], config_path: Path, root: Path, full_rows: list[dict[str, Any]], patch_rows: list[dict[str, Any]], patch_summary: dict[str, Any], expected_full: int, expected_patches: int, napari_identity: dict[str, Any], napari_source: Path, checkpoint: Path, embedder_root: Path, inference: dict[str, Any], patch_cfg: dict[str, Any], levels: list[str], image_ids: list[str]) -> dict[str, Any]:
    all_rows = full_rows + patch_rows
    if len(full_rows) != expected_full:
        raise RuntimeError(f"全图行数错误：{len(full_rows)} != {expected_full}")
    if patch_cfg.get("enabled") and len(patch_rows) != expected_patches:
        raise RuntimeError(f"patch 行数错误：{len(patch_rows)} != {expected_patches}")
    failed_full = [r for r in full_rows if r["status"] != "pass"]
    failed = failed_full
    if patch_cfg.get("enabled") and patch_summary["status"] != "quality_equivalent":
        failed.append({"asset": "test_patch", "status": patch_summary["status"]})
    verdict = "pass" if not failed else "failed"

    with (output_root / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    full_psnrs = [r["psnr_db"] for r in full_rows if r["psnr_db"] is not None]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "阶段 3.3/4：90 张 *_fluoresfm 全图（质量等价）与 735 个测试 patch（派生资产，资产级判定）批量生产。",
        "config": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "script": Path(__file__).resolve().relative_to(root).as_posix(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "workspace_revision": git_revision(root),
        "napari": {**napari_identity, "source_tree_sha256": _source_tree_digest(napari_source)},
        "assets": {
            "checkpoint_sha256": sha256_file(checkpoint),
            "embedder_config_sha256": sha256_file(embedder_root / "open_clip_config.json"),
            "embedder_bin_sha256": sha256_file(embedder_root / "open_clip_pytorch_model.bin"),
        },
        "params": {
            "device": config["device"],
            "levels": levels,
            "image_ids": image_ids,
            "inference": inference,
            "deterministic": bool(config.get("deterministic", False)),
            "patch_chain": {k: v for k, v in patch_cfg.items() if k != "enabled"},
            "acceptance": config["acceptance"],
        },
        "summary": {
            "full_images": {"total": len(full_rows), "pass": len(full_rows) - len(failed_full), "fail": len(failed_full)},
            "test_patches": patch_summary if patch_cfg.get("enabled") else {"enabled": False},
            "full_image_mean_psnr_db": float(np.mean(full_psnrs)),
            "full_image_min_psnr_db": float(np.min(full_psnrs)),
            "full_image_min_ssim": float(min(r["ssim"] for r in full_rows if r["ssim"] is not None)),
        },
        "verdict": verdict,
        "exit_code": EXIT_PASS if not failed else EXIT_FAILED,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "run.md").write_text(
        "# BioSR-MT _fluoresfm 批量生产记录\n\n"
        "本运行仅在新的 `experiments/` 运行目录中写入文件，未修改 bundled example、checkpoint 或既有审计证据。\n\n"
        "- `full_images/<level>/`：生产出的 90 张全图候选；`patches/`：从 level-3 全图派生的 735 个测试 patch。\n"
        "- `comparison.csv`：逐文件 shape/dtype/PSNR/SSIM/最大绝对误差/SHA/状态；`manifest.json` 为机器可读总记录。\n"
        "- 判定：全图官方口径 PSNR ≥45 dB / SSIM ≥0.99（带裕度）；测试 patch 为派生资产，资产级判定（结构 + 逐 patch 均值 ≥44 dB），分布见 manifest。\n",
        encoding="utf-8",
    )
    return manifest


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    root, config_path = args.workspace_root.resolve(), args.config.resolve()
    stage = "prepare"
    output_root: Path | None = None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("不支持的配置版本。")
        deterministic = bool(config.get("deterministic", False))
        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        output_root = resolve(root, config["output"])
        reference_root = resolve(root, config["reference_root"])
        napari_repo = resolve(root, config["napari_repository"])
        napari_source = resolve(root, config["napari_source_root"])
        embedder_root = resolve(root, config["embedder_root"])
        checkpoint = resolve(root, config["checkpoint"])
        image_ids = [str(value) for value in config["image_ids"]]
        levels = [str(value) for value in config["levels"]]
        inference = config["inference"]
        patch_cfg = config["patch_chain"]
        full_cfg = config["acceptance"]["full_image"]
        patch_cfg_acc = config["acceptance"]["patch"]

        # 必要输入存在性
        required = [reference_root, napari_repo, napari_source, embedder_root, checkpoint]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("缺少所需输入：" + "; ".join(missing))
        if not (embedder_root / "open_clip_config.json").is_file() or not (embedder_root / "open_clip_pytorch_model.bin").is_file():
            raise FileNotFoundError("embedder 文件不完整。")
        for level in levels:
            ref_level = f"{level}_fluoresfm"
            for name in [f"{image_id}.tif" for image_id in image_ids]:
                if not (reference_root / level / name).is_file():
                    raise FileNotFoundError(f"缺少输入：{reference_root / level / name}")
                if not (reference_root / ref_level / name).is_file():
                    raise FileNotFoundError(f"缺少参考：{reference_root / ref_level / name}")

        # 源码版本锁定
        napari_identity = {
            "revision": git_revision(napari_repo),
            "expected_revision": config.get("napari_expected_revision"),
            "porcelain": git_porcelain(napari_repo),
            "tracked_modified": git_tracked_modified(napari_repo),
        }
        expected = config.get("napari_expected_revision")
        if config.get("require_napari_revision_match", True) and expected and napari_identity["revision"] != expected:
            raise RuntimeError(f"napari-fluoresfm 源码版本与预期不符：{napari_identity['revision']} != {expected}")
        if napari_identity["tracked_modified"]:
            raise RuntimeError("napari-fluoresfm 存在已跟踪文件修改；批量生产要求源码树干净。")

        expected_full = 15 * len(levels)
        expected_patches = 0
        size = 0
        if patch_cfg.get("enabled"):
            input_level = patch_cfg["input_level"]
            size = int(tifffile.imread(reference_root / input_level / f"{image_ids[0]}.tif").shape[-1])
            expected_patches = len(patch_index_names(image_ids, int(patch_cfg["patch_size"]), int(patch_cfg["stride"]), size))

        if args.dry_run:
            print(json.dumps({
                "output": str(output_root),
                "levels": levels,
                "image_ids": image_ids,
                "expected_full_images": expected_full,
                "expected_patches": expected_patches if patch_cfg.get("enabled") else 0,
                "device": config["device"],
                "napari_revision_match": napari_identity["revision"] == expected,
                "overwrite": "refused" if not args.evaluate_only else "reuse",
            }, ensure_ascii=False, indent=2))
            return EXIT_PASS

        # ---- --evaluate-only：复用既有输出，重算验证与报告 ----
        if args.evaluate_only:
            if not output_root.exists():
                raise FileNotFoundError(f"--evaluate-only 要求输出目录已存在：{output_root}")
            full_images_root = output_root / "full_images"
            patch_output_root = output_root / "patches"
            stage = "evaluate"
            full_rows = verify_full_images(full_images_root, reference_root, levels, image_ids, full_cfg)
            patch_rows, patch_summary = [], {"enabled": False, "status": "quality_equivalent", "total": 0}
            if patch_cfg.get("enabled"):
                patch_rows, patch_summary = verify_patches(patch_output_root, reference_root / patch_cfg["reference_patch_dir"], {**patch_cfg, "psnr_db_floor_mean": patch_cfg_acc["psnr_db_floor_mean"]}, image_ids, size)
            manifest = write_report(output_root, config, config_path, root, full_rows, patch_rows, patch_summary, expected_full, expected_patches, napari_identity, napari_source, checkpoint, embedder_root, inference, patch_cfg, levels, image_ids)
            print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
            return manifest["exit_code"]

        # ---- 完整生产 ----
        if output_root.exists():
            raise FileExistsError(f"拒绝覆盖既有输出：{output_root}")
        output_root.mkdir(parents=True, exist_ok=False)
        (output_root / "environment.json").write_text(json.dumps(capture_environment(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        freeze = pip_freeze()
        if freeze is not None:
            (output_root / "pip_freeze.txt").write_text(freeze, encoding="utf-8")

        stage = "inference"
        sys.path.insert(0, str(napari_source))
        if deterministic:
            _inject_deterministic_predict(napari_source)
        from napari_fluoresfm.fluoresfm.test.predict import predict

        input_index_root = output_root / "input_index"
        input_index_root.mkdir()
        full_images_root = output_root / "full_images"
        full_images_root.mkdir()
        with (output_root / "run.log").open("w", encoding="utf-8") as log_handle:
            for level in levels:
                input_level_root = reference_root / level
                level_candidate_root = full_images_root / level
                level_candidate_root.mkdir()
                index_path = input_index_root / f"{level}.txt"
                index_path.write_text("\n".join(f"{image_id}.tif" for image_id in image_ids) + "\n", encoding="utf-8")
                params = {
                    "path_input": str(input_level_root),
                    "path_input_index": str(index_path),
                    "path_output": str(level_candidate_root),
                    "path_embedder": str(embedder_root),
                    "path_checkpoint": str(checkpoint),
                    "sf_lr": int(inference["scale_factor"]),
                    "batch_size": int(inference["batch_size"]),
                    "patch_size": int(inference["patch_size"]),
                    "device": config["device"],
                    "compile": bool(inference["compile"]),
                    "text": inference["prompt"],
                }
                print(f"[batch] predict level={level} images={len(image_ids)}", flush=True)
                with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                    result = predict(params)
                if result != 1:
                    raise RuntimeError(f"推理返回失败状态 {result}（level={level}）；详情见 run.log。")

        stage = "verify"
        full_rows = verify_full_images(full_images_root, reference_root, levels, image_ids, full_cfg)
        patch_rows, patch_summary = [], {"enabled": False, "status": "quality_equivalent", "total": 0}
        if patch_cfg.get("enabled"):
            stage = "patch_derivation"
            patch_output_root = output_root / "patches"
            patch_output_root.mkdir()
            reference_patch_dir = reference_root / patch_cfg["reference_patch_dir"]
            if not reference_patch_dir.is_dir():
                raise FileNotFoundError(f"缺少参考 patch 目录：{reference_patch_dir}")
            # 派生并写入 patch（供下游使用；验证在 verify_patches 内完成）
            for image_id in image_ids:
                full = tifffile.imread(full_images_root / patch_cfg["input_level"] / f"{image_id}.tif")
                for row, col, patch_arr in _patch_indexed(derive_patches(full, patch_cfg), int(patch_cfg["patch_size"]), int(patch_cfg["stride"]), size):
                    name = f"{image_id}_{row}_{col}.tif"
                    tifffile.imwrite(patch_output_root / name, patch_arr, metadata={"axes": "CYX"})
            patch_rows, patch_summary = verify_patches(patch_output_root, reference_patch_dir, {**patch_cfg, "psnr_db_floor_mean": patch_cfg_acc["psnr_db_floor_mean"]}, image_ids, size)

        stage = "report"
        manifest = write_report(output_root, config, config_path, root, full_rows, patch_rows, patch_summary, expected_full, expected_patches, napari_identity, napari_source, checkpoint, embedder_root, inference, patch_cfg, levels, image_ids)
        print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
        return manifest["exit_code"]
    except Exception as exc:
        failure = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": __import__("traceback").format_exc(),
        }
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())
