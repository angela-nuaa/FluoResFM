#!/usr/bin/env python
"""Validate tracked repository contracts without requiring data or a GPU."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from check_config_path_policy import validate_config


LINK_RE = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
EXTERNALIZED_ROOTS = {"data", "experiments", "outputs"}


def local_link_errors(root: Path) -> list[str]:
    errors: list[str] = []
    documents = [root / "README.md", root / "DATA_AND_MODELS.md", root / "CHANGELOG.md"]
    documents.extend((root / "docs").rglob("*.md"))
    documents.extend((root / "scripts").glob("README.md"))
    documents.extend((root / "assets").glob("README.md"))
    for document in documents:
        if not document.is_file():
            continue
        text = document.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target = match.group(1).split("#", 1)[0].strip("<>")
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            candidate = (document.parent / target).resolve()
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                relative = None
            # Raw inputs and generated run artifacts are intentionally excluded
            # from Git. Documentation may link to their expected local paths;
            # validate those links only in an asset-bearing run, not in CI.
            if relative is not None and relative.parts and relative.parts[0] in EXTERNALIZED_ROOTS:
                continue
            if not candidate.exists():
                errors.append(f"{document.relative_to(root)}: 失效本地链接 {target}")
    return errors


def result_summary_errors(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted((root / "results").glob("*.summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.relative_to(root)}: 无效 JSON：{exc}")
            continue
        for key in ("schema_version", "status", "experiment_id", "scope", "evidence"):
            if key not in payload:
                errors.append(f"{path.relative_to(root)}: 缺少 {key}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    errors = local_link_errors(root)
    errors.extend(error for path in sorted((root / "configs").glob("*.json")) for error in validate_config(path))
    errors.extend(result_summary_errors(root))
    if errors:
        print("repository contract check: FAILED", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("repository contract check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
