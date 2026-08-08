#!/usr/bin/env python
"""Confirm items 1-3 of 推理实验-03: official-vs-current evaluation 口径.

Runs purely on CPU (existing candidates).  Computes vs-SIM / vs-reference
metrics under the official 4_0_result_evaluate.py protocol variants:
  * SIM downsampling method: official interp_sf(sf=-2)=avg_pool2d(k=2) vs
    plain decimation vs resize, to identify what the original B experiment used
    (target: reproduce candidate-vs-SIM mean 35.15 dB).
  * rolling-ball background subtraction (official bkg_subtraction radius=25
    sf=16) applied to ESTIMATED images, which official eval does for
    biosr-mt-dcv-*.  Official impl breaks on 502x502 (bg 496x496 != image);
    we record that and compute a size-fixed variant (reflect-pad to 512).
  * 8-image (41-48) vs 15-image (41-55) means (official num_sample=8).

口径: per-image P3/P99.5 -> clip[0,2.5] -> PSNR/SSIM data_range=2.5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import skimage.io as io
import skimage.restoration
import skimage.transform
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

IDS = list(range(41, 56))  # 41..55
FIRST8 = IDS[:8]

P_LOW, P_HIGH = 0.03, 0.995
CLIP_MAX = 2.5
DATA_RANGE = 2.5


def normalize(img: np.ndarray) -> np.ndarray:
    """Official NormalizePercentile (per-image global), returns same shape."""
    vmin = np.percentile(img, P_LOW * 100)
    vmax = np.percentile(img, P_HIGH * 100)
    if vmax == 0:
        vmax = img.max()
    amp = vmax - vmin
    if amp == 0:
        amp = 1
    return (img - vmin) / amp


def clip025(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, CLIP_MAX)


def prep(img: np.ndarray) -> np.ndarray:
    """Official eval 口径 on one image: P3/P99.5 -> clip[0,2.5], then 2D
    (official passes est[0] / interp_sf(...)[0] to PSNR/SSIM)."""
    out = clip025(normalize(img))
    return out[0] if out.ndim == 3 and out.shape[0] == 1 else out


def avg_pool2d_t(x: np.ndarray, k: int) -> np.ndarray:
    """Official interp_sf(sf=-k): torch avg_pool2d kernel=k stride=k."""
    return torch.nn.functional.avg_pool2d(torch.tensor(x), kernel_size=k, stride=k).numpy()


def interp_sf(x: np.ndarray, sf: int, mode="nearest") -> np.ndarray:
    """Official interp_sf (3D [C,H,W] input). sf>0 upscale, sf<0 avg_pool."""
    t = torch.unsqueeze(torch.tensor(x), dim=0)  # (1,C,H,W)
    if sf > 0:
        out = torch.nn.functional.interpolate(t, scale_factor=sf, mode=mode)
    else:
        out = torch.nn.functional.avg_pool2d(t, kernel_size=-sf, stride=-sf)
    return out[0].numpy()


def rolling_ball_approximation(image: np.ndarray, radius: int, sf: int = 4):
    """Official utils/data.py rolling_ball_approximation."""
    image = np.array(image)
    if len(image.shape) == 2:
        image = image[None]
    image = image.astype(np.float32)
    image_down = interp_sf(image, sf=-sf)
    bg_down = skimage.restoration.rolling_ball(image_down[0], radius=radius)
    bg = interp_sf(bg_down[None], sf=sf, mode="bicubic")
    image_roll = image - bg
    return image_roll, bg


def bkg_subtraction(image: np.ndarray) -> np.ndarray:
    """Official bkg_subtraction (radius=25, sf=16)."""
    image_rb, _ = rolling_ball_approximation(image, radius=25, sf=16)
    return np.clip(image_rb, 0, None)


def bkg_subtraction_size_fixed(image: np.ndarray) -> np.ndarray:
    """Official bkg_subtraction, size-fixed: reflect-pad to multiple of 16,
    run, crop back. Official impl broadcasts bg(496x496) against image(502x502)."""
    H, W = image.shape[-2:]
    if H % 16 == 0 and W % 16 == 0:
        return bkg_subtraction(image)
    pH = (16 - H % 16) % 16
    pW = (16 - W % 16) % 16
    padded = np.pad(image, ((0, 0), (0, pH), (0, pW)), mode="reflect")
    sub, _ = rolling_ball_approximation(padded, radius=25, sf=16)
    return np.clip(sub[..., :H, :W], 0, None)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(a, b, data_range=DATA_RANGE))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, data_range=DATA_RANGE))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs))


def main(workspace_root: Path) -> None:
    cand_root = workspace_root / "experiments/preprocess-03_biosr-mt-fluoresfm-inference-audit/20260807_deconv15_p64"
    inp_root = workspace_root / "example/data/BioSR_MT/test/channel_0/WF_noise_level_3"
    ref_root = workspace_root / "example/data/BioSR_MT/test/channel_0/WF_noise_level_3_fluoresfm"
    sim_root = workspace_root / "example/data/BioSR_MT/test/channel_0/SIM"
    rows: dict[str, list[float]] = {
        "cand_vs_ref_psnr": [], "cand_vs_ref_ssim": [],
        "cand_vs_sim_avgpool_psnr": [], "cand_vs_sim_avgpool_ssim": [],
        "cand_vs_sim_decimate_psnr": [],
        "cand_vs_sim_resize_psnr": [],
        "cand_bkg_vs_sim_avgpool_psnr": [], "cand_bkg_vs_sim_avgpool_ssim": [],
        "ref_vs_sim_avgpool_psnr": [], "ref_vs_sim_avgpool_ssim": [],
        "ref_bkg_vs_sim_avgpool_psnr": [],
        "inp_vs_sim_avgpool_psnr": [],
    }

    official_fails_on_502: list[str] = []

    for i in IDS:
        name = f"{i}.tif"
        cand = io.imread(cand_root / name).astype(np.float32)
        inp = io.imread(inp_root / name).astype(np.float32)
        ref = io.imread(ref_root / name).astype(np.float32)
        sim = io.imread(sim_root / name).astype(np.float32)

        # SIM -> output resolution (official sf_hr=-2 avg_pool2d k=2)
        sim_down = avg_pool2d_t(sim, 2)
        sim_down_dec = sim[:, ::2, ::2]
        sim_down_resize = np.asarray(
            skimage.transform.resize(sim[0], sim_down.shape[-2:], order=3, anti_aliasing=True)
        )[None]

        # ---- official impl failure check on 502 ----
        try:
            _ = bkg_subtraction(cand)
            official_fails_on_502.append(f"{name}:OK")
        except Exception as e:  # noqa: BLE001
            official_fails_on_502.append(f"{name}:{type(e).__name__}")

        # ---- candidate vs bundled reference (A') ----
        a, b = prep(cand), prep(ref)
        rows["cand_vs_ref_psnr"].append(psnr(a, b))
        rows["cand_vs_ref_ssim"].append(ssim(a, b))

        # ---- candidate vs SIM: three downsampling methods ----
        ca, sb = prep(cand), prep(sim_down)
        rows["cand_vs_sim_avgpool_psnr"].append(psnr(ca, sb))
        rows["cand_vs_sim_avgpool_ssim"].append(ssim(ca, sb))
        rows["cand_vs_sim_decimate_psnr"].append(psnr(prep(cand), prep(sim_down_dec)))
        rows["cand_vs_sim_resize_psnr"].append(psnr(prep(cand), prep(sim_down_resize)))

        # ---- candidate(bkg) vs SIM ----
        cand_bkg = bkg_subtraction_size_fixed(cand)
        rows["cand_bkg_vs_sim_avgpool_psnr"].append(psnr(prep(cand_bkg), sb))
        rows["cand_bkg_vs_sim_avgpool_ssim"].append(ssim(prep(cand_bkg), sb))

        # ---- bundled reference vs SIM (B'), with/without bkg ----
        rows["ref_vs_sim_avgpool_psnr"].append(psnr(prep(ref), sb))
        rows["ref_vs_sim_avgpool_ssim"].append(ssim(prep(ref), sb))
        rows["ref_bkg_vs_sim_avgpool_psnr"].append(
            psnr(prep(bkg_subtraction_size_fixed(ref)), sb)
        )

        # ---- WF input vs SIM baseline ----
        rows["inp_vs_sim_avgpool_psnr"].append(psnr(prep(inp), sb))

    print("== official bkg_subtraction on raw (1,502,502):", official_fails_on_502[:3], "...")
    print()
    keys = list(rows)
    print(f"{'metric':34s} {'15-img':>10s} {'first-8':>10s}")
    for k in keys:
        if not rows[k]:
            continue
        m15 = mean(rows[k])
        m8 = mean([v for v, i in zip(rows[k], IDS) if i in FIRST8])
        print(f"{k:34s} {m15:10.4f} {m8:10.4f}")

    print()
    print("== reference points from 推理实验-01/02 (官方口径) ==")
    print("cand vs ref 15-img mean: 48.13 dB / SSIM 0.9974")
    print("cand vs SIM 15-img mean: 35.15 dB (B); inp vs SIM: 27.84 dB")
    print("bundled ref vs SIM 15-img mean: 34.92 dB (B')")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace-root", default=Path.cwd(), type=Path, help="FluoResFM 仓库根目录")
    args = ap.parse_args()
    sys.exit(main(args.workspace_root))
