#!/usr/bin/env python
"""Check that this checkout is ready for a declared FluoResFM workflow.

The command never downloads, modifies, or deletes assets.  Its normal mode
checks the frozen submodule revisions and importable direct runtime modules.
Use ``--assets`` to validate a local asset manifest and ``--production`` to
also require the paths used by the BioSR-MT batch-production configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_REVISIONS = {
    "repos/fluoresfm": "772ac54b2090d4de343979b8ba221c4de541ff85",
    "repos/napari-fluoresfm": "2fded7c31be8c52476de89934ced9093ebe2c307",
}
RUNTIME_MODULES = (
    "numpy",
    "scipy",
    "skimage",
    "tifffile",
    "torch",
    "torchvision",
    "torchaudio",
    "open_clip",
    "transformers",
    "pytorch_msssim",
    "nanopyx",
    "PyQt5",
)
PRODUCTION_PATHS = (
    "example/checkpoints/fluoresfm/epoch_0_iter_700000.pt",
    "example/checkpoints/biomedclip/open_clip_config.json",
    "example/checkpoints/biomedclip/open_clip_pytorch_model.bin",
    "example/data/BioSR_MT/test/channel_0",
)


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_submodules(root: Path) -> list[str]:
    errors: list[str] = []
    for relative, expected in EXPECTED_REVISIONS.items():
        actual = git_revision(root / relative)
        if actual is None:
            errors.append(f"子模块不可用：{relative}（运行 git submodule update --init --recursive）")
        elif actual != expected:
            errors.append(f"子模块版本不匹配：{relative}={actual}，期望 {expected}")
    return errors


def check_runtime() -> list[str]:
    return [f"缺少 Python 模块：{name}" for name in RUNTIME_MODULES if importlib.util.find_spec(name) is None]


def check_asset_manifest(root: Path) -> list[str]:
    path = root / "assets" / "manifest.local.json"
    if not path.is_file():
        return ["缺少 assets/manifest.local.json（从 assets/manifest.local.example.json 复制并填写）。"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"无法读取资产清单：{exc}"]
    if payload.get("schema_version") != "fluoresfm-assets/v1" or not isinstance(payload.get("resources"), list):
        return ["资产清单 schema 无效。"]

    errors: list[str] = []
    for resource in payload["resources"]:
        if not isinstance(resource, dict):
            errors.append("资产清单中存在非对象条目。")
            continue
        label = str(resource.get("id", "<unknown>"))
        archive = resource.get("archive_path")
        expected = resource.get("sha256")
        if not isinstance(archive, str) or not archive:
            errors.append(f"{label}: 缺少 archive_path。")
            continue
        if not isinstance(expected, str) or len(expected) != 64:
            errors.append(f"{label}: 缺少 64 位 SHA-256。")
            continue
        local = root / archive
        if not local.is_file():
            errors.append(f"{label}: 找不到归档 {archive}。")
        elif sha256(local).lower() != expected.lower():
            errors.append(f"{label}: SHA-256 不匹配。")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--assets", action="store_true", help="校验本地资产清单和归档哈希。")
    parser.add_argument("--production", action="store_true", help="检查 BioSR-MT 批量生产所需的已解压资产。")
    parser.add_argument("--skip-runtime", action="store_true", help="仅检查仓库结构，不导入运行时模块。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.workspace_root.resolve()
    errors = check_submodules(root)
    if not args.skip_runtime:
        errors.extend(check_runtime())
    if args.assets:
        errors.extend(check_asset_manifest(root))
    if args.production:
        errors.extend(f"缺少生产输入：{relative}" for relative in PRODUCTION_PATHS if not (root / relative).exists())
    if errors:
        print("doctor: 未就绪", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("doctor: 通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
