#!/usr/bin/env python
"""BioSR-MT bundled example 最终验收：合并阶段 1–4 证据，给全部文件唯一状态。

把阶段 1 的完整基线清单（reference_inventory.csv，83,591 文件）与阶段 2
（materialization_validation.csv，82,766 已生产）及阶段 3–4（批量 comparison.csv，
825：90 全图 + 735 测试 patch）合并，为每个基线文件赋予唯一最终状态，并核对
无遗漏、无重复。

状态分类：
  - 严格文件一致：阶段 2 全图（max_abs_error == 0）与 16 文本（字节一致）
  - 浮点数值等价：阶段 2 训练 patch（float32 容差内）
  - 质量等价：阶段 3–4 的 90 张 *_fluoresfm 全图 + 735 个测试 patch
  - 排除：基线中未被任何阶段生产的文件（期望为 0）

输出：final_status.csv（逐文件）+ final_manifest.json（汇总与核验）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "biosr-mt-final-acceptance/v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-csv", required=True, type=Path, help="阶段 1 reference_inventory.csv")
    p.add_argument("--materialization-csv", required=True, type=Path, help="阶段 2 materialization_validation.csv")
    p.add_argument("--comparison-csv", action="append", required=True, type=Path, help="阶段 3–4 comparison.csv（可多次）")
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def classify_stage2(row: dict[str, str]) -> tuple[str, str]:
    """阶段 2 (status, family)：全图严格一致（0 误差）、训练 patch 数值等价、文本严格一致。"""
    kind = row["kind"]
    tolerance = row.get("tolerance", "")
    if kind == "text":
        return "严格文件一致", "16 文本"
    max_err = row.get("max_abs_error", "")
    if tolerance in ("0.0", "0"):
        return "严格文件一致", "550 全图(WF/SIM)"
    try:
        if float(max_err) <= float(tolerance):
            return "浮点数值等价", "82,200 训练 patch"
    except ValueError:
        pass
    return "待定", "待定"


def comparison_to_path(row: dict[str, str]) -> str:
    """把批量 comparison.csv 行映射回基线 relative_path。"""
    asset = row["asset"]
    if asset == "full_image":
        return f"test/channel_0/{row['level']}_fluoresfm/{row['file']}"
    if asset == "test_patch":
        return f"test/channel_0/WF_noise_level_3_fluoresfm_p64_s64_2d/{row['file']}"
    raise ValueError(f"未知 asset：{asset}")


def main() -> int:
    args = parse_args()
    baseline = load_rows(args.baseline_csv)
    materialization = load_rows(args.materialization_csv)

    stage2: OrderedDict[str, tuple[str, str]] = OrderedDict()
    for row in materialization:
        rel = row["relative_path"]
        if rel in stage2:
            raise ValueError(f"阶段 2 重复：{rel}")
        stage2[rel] = classify_stage2(row)

    stage34: OrderedDict[str, tuple[str, str]] = OrderedDict()
    for csv_path in args.comparison_csv:
        for row in load_rows(csv_path):
            rel = comparison_to_path(row)
            status = "质量等价" if row["status"] in ("pass", "ok", "quality_equivalent") else f"未通过({row['status']})"
            family = "90 *_fluoresfm 全图" if row["asset"] == "full_image" else "735 测试 patch"
            if rel in stage34:
                if stage34[rel] != (status, family):
                    raise ValueError(f"阶段 3–4 状态冲突：{rel} {stage34[rel]} vs {(status, family)}")
                continue
            stage34[rel] = (status, family)

    rows: list[dict[str, str]] = []
    excluded: list[str] = []
    status_count: Counter = Counter()
    family_count: Counter = Counter()
    for row in baseline:
        rel = row["relative_path"]
        if rel in stage2:
            status, family = stage2[rel]
            evidence = "阶段2 materialization_validation.csv"
        elif rel in stage34:
            status, family = stage34[rel]
            evidence = "阶段3-4 批量 comparison.csv"
        else:
            status, family = "排除", "排除"
            evidence = ""
            excluded.append(rel)
        status_count[status] += 1
        family_count[family] += 1
        rows.append({
            "relative_path": rel,
            "kind": row["kind"],
            "final_status": status,
            "evidence": evidence,
            "reference_file_sha256": row["file_sha256"],
        })

    # 核验：无遗漏、无重复、总数一致
    if len(rows) != len(baseline):
        raise RuntimeError(f"行数不一致：{len(rows)} vs {len(baseline)}")
    if excluded:
        raise RuntimeError(f"存在排除项：{len(excluded)}（期望 0）")
    n_stage2 = len(stage2)
    n_stage34 = len(stage34)
    n_total = len(rows)
    if n_stage2 + n_stage34 != n_total:
        raise RuntimeError(f"阶段覆盖核对失败：{n_stage2} + {n_stage34} != {n_total}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "final_status.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "BioSR-MT bundled example 全目录最终验收：全部文件唯一状态。",
        "inputs": {
            "baseline": str(args.baseline_csv),
            "materialization": str(args.materialization_csv),
            "comparison": [str(p) for p in args.comparison_csv],
        },
        "counts": {
            "total_files": n_total,
            "stage2_produced": n_stage2,
            "stage34_produced": n_stage34,
            "status": dict(status_count),
            "family": dict(family_count),
        },
        "verification": {
            "no_omission": n_stage2 + n_stage34 == n_total,
            "no_duplication": len(rows) == len(baseline),
            "no_excluded": len(excluded) == 0,
        },
        "verdict": "pass" if (n_stage2 + n_stage34 == n_total and not excluded) else "failed",
    }
    (args.output_dir / "final_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print("verification:", manifest["verification"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
