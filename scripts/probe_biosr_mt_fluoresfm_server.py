#!/usr/bin/env python
"""Run one restricted-server single-image FluoResFM inference probe.

This is the Stage 3 single-image probe of the BioSR-MT strict-production
plan (``docs/plans/计划_BioSR-MT全目录严格生产.md``).  It runs on a
restricted GPU server, freezes the runtime environment and the
napari-fluoresfm source/checkpoint identity, isolates a single input TIFF in
the new run directory, executes the pinned upstream ``predict`` path, and
compares the candidate against one bundled ``*_fluoresfm`` reference with
shape/dtype/array/SHA-256 rules.

Exit codes: 0 = strict (byte-identical), 1 = numerically equivalent only,
2 = comparison failed, 3 = runtime/preparation error.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import json
import platform
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


SCHEMA_VERSION = "biosr-mt-fluoresfm-server-probe/v1"
EXIT_STRICT = 0
EXIT_NUMERIC = 1
EXIT_FAILED = 2
EXIT_RUNTIME = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace-root", default=Path.cwd(), type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and inputs, print the planned scope, and exit without writing output.")
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


def git_tracked_modified(path: Path) -> list[str]:
    try:
        output = subprocess.check_output(["git", "-C", str(path), "diff", "--name-only"], text=True, stderr=subprocess.DEVNULL).strip()
        return [line for line in output.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        return ["<git unavailable>"]


def git_porcelain(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def source_tree_digest(root: Path) -> str:
    """Deterministic SHA-256 over every file under ``root``."""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and "__pycache__" not in item.parts):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def capture_nvidia_smi() -> list[dict[str, str]] | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True, timeout=30, stderr=subprocess.DEVNULL,
        )
        rows = []
        for line in output.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2:
                rows.append({"name": parts[0], "driver_version": parts[1]})
        return rows or None
    except (OSError, subprocess.SubprocessError):
        return None


def capture_import_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def capture_environment() -> dict[str, Any]:
    import torch

    environment: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "devices": [],
        "driver": capture_nvidia_smi(),
        "triton_version": capture_import_version("triton"),
        "numpy_version": np.__version__,
        "tifffile_version": tifffile.__version__,
    }
    if torch.cuda.is_available():
        environment["devices"] = [
            {"index": index, "name": torch.cuda.get_device_name(index), "capability": list(torch.cuda.get_device_capability(index))}
            for index in range(torch.cuda.device_count())
        ]
    return environment


def pip_freeze() -> str | None:
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=120, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None


def parse_upstream_behaviors(source: str) -> dict[str, Any]:
    def find(pattern: str) -> str | None:
        match = re.search(pattern, source)
        return match.group(1) if match else None

    return {
        "enable_amp": find(r'"enable_amp":\s*(True|False)'),
        "compile_model_default": find(r'"complie_model":\s*(True|False)'),
        "float32_matmul_precision": find(r'torch\.set_float32_matmul_precision\("([^"]+)"\)'),
        "cudnn_benchmark": find(r"torch\.backends\.cudnn\.benchmark\s*=\s*(True|False)"),
        "percentiles": find(r'"percentiles":\s*\(([^)]*)\)'),
    }


def load_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Path], list[str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("不支持的配置版本。")
    keys = ["input_root", "reference_root", "napari_source_root", "napari_repository", "embedder_root", "checkpoint"]
    resolved = {key: resolve(root, config[key]) for key in keys}
    missing = [str(path) for path in resolved.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少所需输入：" + "; ".join(missing))
    image_ids = [str(value) for value in config["image_ids"]]
    names = [f"{image_id}.tif" for image_id in image_ids]
    if not all((resolved["input_root"] / name).is_file() and (resolved["reference_root"] / name).is_file() for name in names):
        raise FileNotFoundError("输入或参考 TIFF 不完整。")
    if not (resolved["embedder_root"] / "open_clip_config.json").is_file() or not (resolved["embedder_root"] / "open_clip_pytorch_model.bin").is_file():
        raise FileNotFoundError("embedder 文件不完整。")
    return config, resolved, names


def revision_identity(repository: Path, expected: str | None) -> dict[str, Any]:
    actual = git_revision(repository)
    modified = git_tracked_modified(repository)
    return {
        "revision": actual,
        "expected_revision": expected,
        "revision_match": bool(expected and actual == expected),
        "tracked_modified": modified,
        "porcelain": git_porcelain(repository),
    }


def main() -> int:
    args = parse_args()
    root, config_path = args.workspace_root.resolve(), args.config.resolve()
    stage = "prepare"
    output_root: Path | None = None
    output_created = False
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        output_root = resolve(root, raw_config["output"])
        if output_root.exists() and not args.dry_run:
            raise FileExistsError(f"拒绝覆盖既有输出：{output_root}")
        config, resolved, names = load_config(config_path, root)

        napari_identity = revision_identity(resolved["napari_repository"], config.get("napari_expected_revision"))
        if args.dry_run:
            print(json.dumps(
                {
                    "output": str(output_root),
                    "image_ids": [str(value) for value in config["image_ids"]],
                    "device": config["device"],
                    "scale_factor": int(config["scale_factor"]),
                    "batch_size": int(config["batch_size"]),
                    "patch_size": int(config["patch_size"]),
                    "compile": bool(config["compile"]),
                    "napari_revision": napari_identity["revision"],
                    "napari_expected_revision": napari_identity["expected_revision"],
                    "napari_revision_match": napari_identity["revision_match"],
                    "napari_tracked_modified": napari_identity["tracked_modified"],
                    "overwrite": "refused",
                },
                ensure_ascii=False, indent=2,
            ))
            return EXIT_STRICT

        expected = config.get("napari_expected_revision")
        require_match = bool(config.get("require_napari_revision_match", True))
        if require_match and expected and not napari_identity["revision_match"]:
            raise RuntimeError(f"napari-fluoresfm 源码版本与预期不符：{napari_identity['revision']} != {expected}；请先 checkout 到历史候选提交。")
        if napari_identity["tracked_modified"]:
            raise RuntimeError("napari-fluoresfm 存在已跟踪文件修改；严格探针要求源码树干净。")

        output_root.mkdir(parents=True, exist_ok=False)
        output_created = True
        probe_input_root = output_root / "probe_input"
        probe_input_root.mkdir()
        candidate_root = output_root / "candidate"
        candidate_root.mkdir()
        index_path = output_root / "input_index.txt"
        index_path.write_text("\n".join(names) + "\n", encoding="utf-8")

        stage = "probe_input"
        input_hashes = {}
        for name in names:
            source = resolved["input_root"] / name
            copied = probe_input_root / name
            shutil.copy2(source, copied)
            if sha256_file(source) != sha256_file(copied):
                raise RuntimeError(f"探针输入复制后哈希不一致：{name}")
            input_hashes[name] = sha256_file(copied)

        stage = "environment_freeze"
        environment = capture_environment()
        if config["device"].startswith("cuda") and not environment["cuda_available"]:
            raise RuntimeError("配置要求 CUDA 设备，但当前环境不可用。")
        (output_root / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        freeze = pip_freeze()
        if freeze is not None:
            (output_root / "pip_freeze.txt").write_text(freeze, encoding="utf-8")

        stage = "source_lock"
        source_root = resolved["napari_source_root"]
        predict_source = (source_root / "napari_fluoresfm" / "fluoresfm" / "test" / "predict.py").read_text(encoding="utf-8")
        upstream_behaviors = parse_upstream_behaviors(predict_source)
        assets = {
            "input_hashes": input_hashes,
            "checkpoint_sha256": sha256_file(resolved["checkpoint"]),
            "embedder_config_sha256": sha256_file(resolved["embedder_root"] / "open_clip_config.json"),
            "embedder_bin_sha256": sha256_file(resolved["embedder_root"] / "open_clip_pytorch_model.bin"),
        }

        stage = "inference"
        sys.path.insert(0, str(source_root))
        from napari_fluoresfm.fluoresfm.test.predict import predict

        params = {
            "path_input": str(probe_input_root),
            "path_input_index": str(index_path),
            "path_output": str(candidate_root),
            "path_embedder": str(resolved["embedder_root"]),
            "path_checkpoint": str(resolved["checkpoint"]),
            "sf_lr": int(config["scale_factor"]),
            "batch_size": int(config["batch_size"]),
            "patch_size": int(config["patch_size"]),
            "device": config["device"],
            "compile": bool(config["compile"]),
            "text": config["prompt"],
        }
        with (output_root / "run.log").open("w", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                result = predict(params)
        if result != 1:
            raise RuntimeError(f"上游推理返回失败状态：{result}；详情见 run.log。")

        stage = "comparison"
        tolerance = float(config["acceptance"]["numeric_tolerance"])
        rows: list[dict[str, Any]] = []
        for name in names:
            candidate, reference = candidate_root / name, resolved["reference_root"] / name
            if not candidate.is_file():
                raise FileNotFoundError(f"推理未生成：{candidate}")
            candidate_sha, reference_sha = sha256_file(candidate), sha256_file(reference)
            actual, expected = tifffile.imread(candidate), tifffile.imread(reference)
            compatible = actual.shape == expected.shape and actual.dtype == expected.dtype
            max_error = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))) if compatible else None
            mean_error = float(np.mean(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))) if compatible else None
            if candidate_sha == reference_sha:
                status = "strict"
            elif compatible and max_error is not None and max_error <= tolerance:
                status = "numerically_equivalent"
            else:
                status = "failed"
            rows.append({
                "file": name,
                "shape_match": actual.shape == expected.shape,
                "dtype_match": actual.dtype == expected.dtype,
                "max_abs_error": max_error,
                "mean_abs_error": mean_error,
                "candidate_sha256": candidate_sha,
                "reference_sha256": reference_sha,
                "status": status,
            })
        with (output_root / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        statuses = [row["status"] for row in rows]
        if any(status == "failed" for status in statuses):
            verdict, exit_code = "failed", EXIT_FAILED
        elif any(status == "numerically_equivalent" for status in statuses):
            verdict, exit_code = "numerically_equivalent", EXIT_NUMERIC
        else:
            verdict, exit_code = "strict", EXIT_STRICT

        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "阶段 3 单图服务器探针：隔离单张输入并运行固定版本上游推理，比对 bundled *_fluoresfm 参考；不覆盖 bundled example。",
            "config": config_path.relative_to(root).as_posix(),
            "config_sha256": sha256_file(config_path),
            "script": Path(__file__).resolve().relative_to(root).as_posix(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "command_line": sys.argv,
            "workspace_revision": git_revision(root),
            "napari": {**napari_identity, "source_tree_sha256": source_tree_digest(source_root), "predict_py_sha256": sha256_file(source_root / "napari_fluoresfm" / "fluoresfm" / "test" / "predict.py")},
            "assets": assets,
            "environment": environment,
            "upstream_source_behaviors": upstream_behaviors,
            "params": params,
            "acceptance": {"numeric_tolerance": tolerance},
            "comparison": rows,
            "verdict": verdict,
            "exit_code": exit_code,
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_root / "run.md").write_text(
            "# 阶段 3 服务器单图探针记录\n\n"
            "本运行仅在新的 `experiments/` 运行目录中写入文件，未修改 bundled example、checkpoint 或既有审计证据。\n\n"
            "- `probe_input/`：仅含本次单图输入（复制并校验哈希）。\n"
            "- `candidate/`：固定版本上游推理输出。\n"
            "- `environment.json` 与 `pip_freeze.txt`：服务器环境冻结。\n"
            "- `comparison.csv`：逐文件 shape/dtype/数组/SHA-256 比较；`manifest.json` 为机器可读总记录。\n",
            encoding="utf-8",
        )
        print(json.dumps({"verdict": verdict, "exit_code": exit_code, "comparison": rows}, ensure_ascii=False, indent=2))
        return exit_code
    except Exception as exc:
        failure = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if output_root is not None and (output_created or not output_root.exists()):
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())
