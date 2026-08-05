"""Validate the repository's machine-readable configuration naming policy."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
EXPERIMENT_ID_RE = re.compile(r"^[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9])?$")
OUTPUT_PATH_RE = re.compile(r"^(?:experiments|data/derived)/[A-Za-z0-9][A-Za-z0-9._/-]*$")


def validate_config(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: 无法读取 JSON：{exc}"]
    if not isinstance(config, dict):
        return [f"{path}: 顶层必须是对象。"]

    def walk(value: Any, location: str = "$") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str) or not KEY_RE.fullmatch(key):
                    errors.append(f"{path}:{location} 的键 {key!r} 必须为 ASCII lower_snake_case。")
                walk(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]")

    walk(config)
    experiment_id = config.get("experiment_id")
    if not isinstance(experiment_id, str) or not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        errors.append(f"{path}: experiment_id 必须为稳定的 ASCII slug。")
    if not isinstance(config.get("experiment_label"), str) or not config["experiment_label"].strip():
        errors.append(f"{path}: experiment_label 必须为非空展示文本。")

    def check_output(value: Any, location: str) -> None:
        if not isinstance(value, str) or not OUTPUT_PATH_RE.fullmatch(value):
            errors.append(f"{path}:{location} 必须是以 experiments/ 或 data/derived/ 开头的 ASCII 相对输出路径。")

    if isinstance(config.get("output"), str):
        check_output(config["output"], "$.output")
    elif isinstance(config.get("output"), dict) and "root" in config["output"]:
        check_output(config["output"]["root"], "$.output.root")
    elif "output_json" in config:
        check_output(config["output_json"], "$.output_json")
    else:
        errors.append(f"{path}: 缺少 output、output.root 或 output_json。")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 configs/ 的键名和项目自建输出路径策略。")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    args = parser.parse_args()
    paths = sorted(args.config_dir.rglob("*.json"))
    if not paths:
        print(f"未找到 JSON 配置：{args.config_dir}", file=sys.stderr)
        return 2
    errors = [error for path in paths for error in validate_config(path)]
    if errors:
        print("配置路径策略检查失败：", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"配置路径策略检查通过：{len(paths)} 个 JSON 配置。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
