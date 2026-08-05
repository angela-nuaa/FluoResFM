#!/usr/bin/env python
"""审计 BioTISR-CCP 原始 MRC 到 bundled example 微调入口的一致性。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import tifffile


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_reader(path: Path):
    spec = importlib.util.spec_from_file_location("biotisr_mrc_reader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 MRC 读取器：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_mrc


def as_tyx(stack: np.ndarray) -> np.ndarray:
    return np.moveaxis(stack, -1, 0).transpose(0, 2, 1).astype(np.float32)


def normalize(image: np.ndarray, lower: float, upper: float) -> np.ndarray:
    image = np.clip(image.astype(np.float32), 0, None)
    lo, hi = np.percentile(image, (lower, upper))
    return (image - lo) / (hi - lo if hi > lo else 1.0)


def error(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"最大绝对误差": float(delta.max()), "平均绝对误差": float(delta.mean())}


def summary(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        "最大绝对误差": max(value["最大绝对误差"] for value in values),
        "平均绝对误差": float(np.mean([value["平均绝对误差"] for value in values])),
    }


def main() -> None:
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "biotisr-ccp-example-equivalence/v1":
        raise ValueError("不支持的配置版本。")
    raw_root = Path(config["raw_data_root"])
    example_root = Path(config["example_root"])
    cell = config["cell"]
    frames = int(config["raw_frames_per_timepoint"])
    timepoints = int(config["n_timepoints"])
    atol = float(config["float_patch_tolerance"])
    reader = load_reader(Path(config["mrc_reader"]))
    cell_root = raw_root / cell

    report: dict[str, object] = {
        "审计范围": "bundled example 的 Cell_001；不代表论文完整数据切分或其余原始细胞。",
        "细胞": cell,
        "规则": {
            "RawSIMData_level_01/02/03": "轴转换后按连续 9 帧分组，逐像素算术平均，分别得到 WF_noise_level_0/1/2。",
            "SIM_gt": "轴转换后逐像素截断负值为 0，得到 SIM。",
            "SR 微调补丁": "Cell_001 时间点 0：输入 P3/P99.5 标准化后 16×16 行优先 p32/s32；靶标同样标准化后 16×16 行优先 p64/s64。",
        },
        "完整帧": {},
    }
    wf_t0: np.ndarray | None = None
    for raw_level in range(1, 4):
        _, xyz = reader(str(cell_root / f"RawSIMData_level_{raw_level:02d}.mrc"))
        calculated = as_tyx(xyz).reshape(timepoints, frames, 512, 512).mean(axis=1)
        name = f"WF_noise_level_{raw_level - 1}"
        expected = np.stack([
            tifffile.imread(example_root / name / f"{cell}_{t}.tif").squeeze()
            for t in range(timepoints)
        ]).astype(np.float32)
        item = error(calculated, expected)
        item["逐像素完全相等"] = bool(item["最大绝对误差"] == 0.0)
        report["完整帧"][name] = item
        if raw_level == 1:
            wf_t0 = calculated[0]

    _, gt_xyz = reader(str(cell_root / "SIM_gt.mrc"))
    sim = np.clip(as_tyx(gt_xyz), 0, None)
    expected_sim = np.stack([
        tifffile.imread(example_root / "SIM" / f"{cell}_{t}.tif").squeeze()
        for t in range(timepoints)
    ]).astype(np.float32)
    sim_item = error(sim, expected_sim)
    sim_item["逐像素完全相等"] = bool(sim_item["最大绝对误差"] == 0.0)
    report["完整帧"]["SIM"] = sim_item

    assert wf_t0 is not None
    lower, upper = config["patch_normalization_percentiles"]
    source = normalize(wf_t0, lower, upper)
    target = normalize(sim[0], lower, upper)
    lr_errors: list[dict[str, float]] = []
    hr_errors: list[dict[str, float]] = []
    lr_names: list[str] = []
    hr_names: list[str] = []
    for row in range(16):
        for col in range(16):
            name = f"{cell}_0_{row}_{col}.tif"
            lr_names.append(name)
            expected = tifffile.imread(example_root / "WF_noise_level_0_0_p32_s32_2d" / name)
            lr_errors.append(error(source[row * 32:(row + 1) * 32, col * 32:(col + 1) * 32], expected))
    for row in range(16):
        for col in range(16):
            name = f"{cell}_0_{row}_{col}.tif"
            hr_names.append(name)
            expected = tifffile.imread(example_root / "SIM_0_p64_s64_2d" / name)
            hr_errors.append(error(target[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64], expected))
    actual_lr_names = {path.name for path in (example_root / "WF_noise_level_0_0_p32_s32_2d").glob("*.tif")}
    actual_hr_names = {path.name for path in (example_root / "SIM_0_p64_s64_2d").glob("*.tif")}
    patch = {
        "输入补丁": summary(lr_errors),
        "靶标补丁": summary(hr_errors),
        "输入文件集合完全一致": actual_lr_names == set(lr_names),
        "靶标文件集合完全一致": actual_hr_names == set(hr_names),
        "补丁总数": {"输入": len(lr_errors), "靶标": len(hr_errors)},
    }
    patch["在容差内一致"] = bool(
        patch["输入补丁"]["最大绝对误差"] <= atol
        and patch["靶标补丁"]["最大绝对误差"] <= atol
        and patch["输入文件集合完全一致"] and patch["靶标文件集合完全一致"]
    )
    report["SR微调补丁"] = patch
    report["结论"] = {
        "bundled_example入口完全复现": bool(
            all(item["逐像素完全相等"] for item in report["完整帧"].values())
            and patch["在容差内一致"]
        ),
        "说明": "完整帧为严格零误差；TIFF 补丁因 float32 序列化以配置容差判定。",
    }
    output = Path(config["output_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["结论"]["bundled_example入口完全复现"]:
        raise SystemExit("BioTISR-CCP bundled example 一致性审计失败。")


if __name__ == "__main__":
    main()
