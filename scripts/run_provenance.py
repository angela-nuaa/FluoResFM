"""Small, dependency-light helpers for auditable local experiment records."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Mapping


def git_revision(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_info() -> dict[str, str | None]:
    info: dict[str, str | None] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": None,
        "cuda": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def provenance(
    *, root: Path, script: Path, assets: Mapping[str, Path], parameters: Mapping[str, object]
) -> dict[str, object]:
    """Return JSON-safe provenance without copying assets into the repository."""
    missing = [name for name, path in assets.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot record missing file asset(s): {', '.join(missing)}")
    return {
        "root_commit": git_revision(root),
        "napari_fluoresfm_commit": git_revision(root / "repos" / "napari-fluoresfm"),
        "fluoresfm_commit": git_revision(root / "repos" / "fluoresfm"),
        "script": str(script.resolve()),
        "script_sha256": sha256_file(script),
        "command": list(sys.argv),
        "runtime": runtime_info(),
        "assets": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in assets.items()
        },
        "parameters": {name: str(value) for name, value in parameters.items()},
    }


def write_run_md(path: Path, manifest: Mapping[str, object]) -> None:
    record = manifest["provenance"]
    runtime = record["runtime"]
    lines = [
        "# Run record",
        "",
        f"- Created (UTC): {manifest['created_utc']}",
        f"- Scope: {manifest.get('scope', manifest.get('experiment', 'unspecified'))}",
        f"- Root commit: {record['root_commit']}",
        f"- FluoResFM commit: {record['fluoresfm_commit']}",
        f"- napari-fluoresfm commit: {record['napari_fluoresfm_commit']}",
        f"- Python: {runtime['python']}",
        f"- PyTorch / CUDA: {runtime['torch']} / {runtime['cuda']}",
        "- Command: `" + " ".join(record["command"]) + "`",
        "",
        "The authoritative machine-readable record is `manifest.json`. Assets and outputs are local-only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
