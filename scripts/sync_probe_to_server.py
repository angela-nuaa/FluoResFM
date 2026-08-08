#!/usr/bin/env python
"""One-way push of Stage 3 probe assets to a restricted server workspace.

Transfers only audited probe artifacts into the user's personal remote
directory.  Prefers rsync (WSL, then native) and falls back to OpenSSH scp.
rsync ``-z`` compresses on the wire while remote files stay byte-identical,
so local SHA-256 values remain valid on the server.  Never deletes remote
content and never stores credentials (use ssh-agent or let the transfer tool
prompt).

Modes:
  default               small files + napari-fluoresfm submodule (with .git)
  --include-probe-data  + probe weights + the 2 probe TIFFs (~3.4 GB)
  --include-data        + full example/data/BioSR_MT + example/checkpoints (~4.9 GB)
  --force-scp           use OpenSSH scp directly (e.g. when WSL cannot
                        resolve a Windows-only SSH config alias)

Examples:
  python scripts/sync_probe_to_server.py --host gpu01 --user lc --dry-run
  python scripts/sync_probe_to_server.py --host gpu01 --user lc --include-probe-data
  python scripts/sync_probe_to_server.py --host gpu01 --user lc --include-data
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_REMOTE_DIR = "~/lc/FluoResFM"

PROBE_FILES = [
    "configs/biosr_mt_fluoresfm_server_probe_20260805_s1.json",
    "configs/biosr_mt_fluoresfm_server_probe_20260805_s2.json",
    "scripts/probe_biosr_mt_fluoresfm_server.py",
    "scripts/sync_probe_to_server.py",
    "docs/plans/计划_BioSR-MT全目录严格生产.md",
]

SUBMODULE_DIR = "repos/napari-fluoresfm"

PROBE_WEIGHTS = [
    "example/checkpoints/fluoresfm/epoch_0_iter_700000.pt",
    "example/checkpoints/biomedclip/open_clip_config.json",
    "example/checkpoints/biomedclip/open_clip_pytorch_model.bin",
]

PROBE_TIFFS = [
    "example/data/BioSR_MT/test/channel_0/WF_noise_level_3/41.tif",
    "example/data/BioSR_MT/test/channel_0/WF_noise_level_3_fluoresfm/41.tif",
]

FULL_DATA_DIRS = [
    "example/data/BioSR_MT",
    "example/checkpoints",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="云端主机名或 IP")
    parser.add_argument("--user", required=True, help="云端受限普通账户")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR, help=f"云端个人目录（默认 {DEFAULT_REMOTE_DIR}；只允许个人目录，禁止共享路径）")
    parser.add_argument("--port", default=22, type=int, help="SSH 端口（默认 22）")
    parser.add_argument("--include-probe-data", action="store_true", help="同时同步探针最小数据：3 个权重文件 + 2 个探针 TIFF（约 3.4 GB）")
    parser.add_argument("--include-data", action="store_true", help="同时同步完整 example/data/BioSR_MT 与 example/checkpoints（约 4.9 GB，批量/基线用）")
    parser.add_argument("--force-scp", action="store_true", help="跳过 rsync 探测，强制使用 OpenSSH scp")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令与传输工具，不传输")
    return parser.parse_args()


def detect_transfer(force_scp: bool) -> tuple[str, list[str]]:
    if force_scp:
        if not shutil.which("scp"):
            raise RuntimeError("未找到 scp 传输工具。")
        return "scp", []
    if shutil.which("rsync"):
        return "rsync-native", []
    if shutil.which("wsl"):
        try:
            subprocess.run(["wsl", "rsync", "--version"], capture_output=True, timeout=30, check=True)
            return "rsync-wsl", ["wsl"]
        except (OSError, subprocess.SubprocessError):
            pass
    if shutil.which("scp"):
        return "scp", []
    raise RuntimeError("未找到 rsync 或 scp 传输工具。")


def wsl_path(source: Path) -> str:
    text = str(source.resolve())
    drive, rest = text.split(":", 1)
    return "/mnt/" + drive.lower() + rest.replace("\\", "/")


def build_commands(root: Path, args: argparse.Namespace, kind: str, prefix: list[str]) -> list[list[str]]:
    remote_base = args.remote_dir.rstrip("/")
    files: list[tuple[Path, str]] = []
    dirs: list[tuple[Path, str]] = []

    for relative in PROBE_FILES:
        files.append((root / relative, f"{remote_base}/{relative}"))
    dirs.append((root / SUBMODULE_DIR, f"{remote_base}/repos/"))

    if args.include_data:
        for relative in FULL_DATA_DIRS:
            dirs.append((root / relative, f"{remote_base}/{Path(relative).parent.as_posix()}/"))
    elif args.include_probe_data:
        for relative in PROBE_WEIGHTS + PROBE_TIFFS:
            files.append((root / relative, f"{remote_base}/{relative}"))

    missing = [str(source) for source, _ in files + dirs if not source.exists()]
    if missing:
        raise FileNotFoundError("缺少待同步路径：" + "; ".join(missing))

    commands: list[list[str]] = []
    wsl_cache: dict[Path, str] = {}
    if kind.startswith("rsync"):
        for source, target in files + dirs:
            src = str(source)
            if kind == "rsync-wsl":
                if source not in wsl_cache:
                    wsl_cache[source] = wsl_path(source)
                src = wsl_cache[source]
            command = [*prefix, "rsync", "-avz"]
            if source.is_dir():
                command += ["--exclude", "__pycache__"]
            command += ["-e", f"ssh -p {args.port}", src, f"{args.user}@{args.host}:{target}"]
            commands.append(command)
    else:
        parents = sorted({str(Path(target).parent).replace("\\", "/") for _, target in files} | {target.rstrip("/") for _, target in dirs})
        commands.append(["ssh", "-p", str(args.port), f"{args.user}@{args.host}", "mkdir", "-p", *parents])
        for source, target in files + dirs:
            commands.append(["scp", "-C", "-P", str(args.port), "-r", str(source), f"{args.user}@{args.host}:{target}"])
    return commands


def main() -> int:
    args = parse_args()
    if not (args.remote_dir.startswith("~/") or args.remote_dir.startswith("/")):
        print(f"[WARNING] --remote-dir 不是家目录/绝对路径：{args.remote_dir}；请确认它位于你的个人目录。", file=sys.stderr)
    kind, prefix = detect_transfer(args.force_scp)
    root = Path.cwd()
    commands = build_commands(root, args, kind, prefix)
    print(f"# 传输工具：{kind}")
    print(f"# 远端目录：{args.remote_dir}")
    for command in commands:
        print(" ".join(command))
    if not args.dry_run:
        for command in commands:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
