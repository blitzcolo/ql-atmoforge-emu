#!/usr/bin/env python3
"""生成与 `ql-atmoforge merge` 产出结构完全一致的合成数据集（冒烟测试用）。

真实数据未生成前，用它验证 prepare_data / train / evaluate 全链路。
光谱由参数的光滑解析函数合成（Beer–Lambert + Planck 形状，非真实物理），
并按真实数据的方式注入 failed（全零 + status=2）、partial（status=1，部分含
NaN 列）与整谱饱和样本（浓雾 × 长路径 → TOTAL 四位小数打印为 0）。

  python scripts/make_fake_data.py --out tmp/fake_mwir_slant --n 2048
  python scripts/make_fake_data.py --out tmp/fake_mwir_slant_test --n 512 \
      --sampler random --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atmoemu.manifest import EARTH_RADIUS_KM, slant_range_km  # noqa: E402

C1 = 1.191042972e-12
C2 = 1.4387769


def planck(nu, T):
    return C1 * nu ** 3 / np.expm1(C2 * nu / T)


def gauss(nu, c, w):
    return np.exp(-0.5 * ((nu - c) / w) ** 2)


BANDS = {
    # 缩小 K 的玩具波段；结构与真实 preset 一致
    "mwir_toy": {"name": "mwir", "v1_cm": 2000.0, "v2_cm": 2095.0,
                 "dv_cm": 1.0, "fwhm_cm": 2.0, "K": 96, "thermal": True},
    "swir_toy": {"name": "swir", "v1_cm": 4167.0, "v2_cm": 4246.0,
                 "dv_cm": 1.0, "fwhm_cm": 2.0, "K": 80, "thermal": False},
}

TAU_COLS = ["TOTAL", "LOG_TOTAL"]


def sampled_spec(path_type: str) -> dict:
    s = {
        "atmos_model": {"values": [2, 3]},
        "ihaze": {"values": [1, 4, 5, 9, 10]},
        "icld": {"values": [0, 6, 18]},
        "vis_km": {"log_uniform": [0.5, 50.0]},
        "rainrt_mm_h": {"uniform": [0.0, 50.0]},
        "t_ground_K": {"uniform": [253.0, 328.0]},
        "rh": {"uniform": [0.05, 1.0]},
        "p_hPa": {"uniform": [950.0, 1040.0]},
        "h2o_scale": {"uniform": [0.5, 2.0]},
    }
    if path_type == "slant_to_ground":
        s["h1_km"] = {"uniform": [0.5, 12.0]}
        s["view_zenith_deg"] = {"uniform": [110.0, 180.0]}
    else:
        s["h1_km"] = {"uniform": [0.0, 0.5]}
        s["range_km"] = {"log_uniform": [0.05, 20.0]}
    s["sun_zenith_deg"] = {"uniform": [0.0, 85.0]}
    s["sun_rel_azimuth_deg"] = {"uniform": [0.0, 180.0]}
    return s


def draw(rng, dist, n):
    if "values" in dist:
        return rng.choice(np.asarray(dist["values"], dtype=np.float64), size=n)
    if "uniform" in dist:
        lo, hi = dist["uniform"]
        return rng.uniform(lo, hi, n)
    lo, hi = dist["log_uniform"]
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sampler", default="sobol", choices=["sobol", "random"])
    ap.add_argument("--band", default="mwir_toy", choices=list(BANDS))
    ap.add_argument("--path", default="slant_to_ground",
                    choices=["slant_to_ground", "horizontal"])
    args = ap.parse_args()

    band = BANDS[args.band]
    thermal = band["thermal"]
    path_type = args.path
    K, N = band["K"], args.n
    nu = band["v1_cm"] + band["dv_cm"] * np.arange(K)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = sampled_spec(path_type)
    cols = {name: draw(rng, dist, N) for name, dist in spec.items()}

    # resolve（与生成端 ParamSpace::resolve 一致）
    cols["rainrt_mm_h"] = np.where(cols["icld"] == 6, cols["rainrt_mm_h"], 0.0)
    h1 = cols["h1_km"]
    if path_type == "horizontal":
        view = np.full(N, 90.0)
        h2 = h1.copy()
        rng_km = cols["range_km"]
    else:
        view = cols["view_zenith_deg"]
        h2 = np.zeros(N)
        rng_km = np.array([slant_range_km(a, b) for a, b in zip(h1, view)])
    cosv = np.cos(np.radians(view))

    feature_names = list(spec.keys()) + ["h1_km", "h2_km",
                                         "cos_view_zenith", "range_km"]
    params = np.stack([cols[n] for n in spec] + [h1, h2, cosv, rng_km],
                      axis=1).astype(np.float64)

    # -------------------------------------------------------- 合成光谱 --
    g_h2o = gauss(nu, band["v1_cm"] + 0.25 * (band["v2_cm"] - band["v1_cm"]), 12.0)
    g_gas = gauss(nu, band["v1_cm"] + 0.65 * (band["v2_cm"] - band["v1_cm"]), 7.0)
    slope = (nu - band["v1_cm"]) / (band["v2_cm"] - band["v1_cm"])

    if path_type == "horizontal":
        pathfac = rng_km * np.exp(-h1 / 8.0)
    else:
        pathfac = 8.0 * (1.0 - np.exp(-h1 / 8.0)) / np.maximum(-cosv, 0.05)
    pf = pathfac[:, None]
    k_ext = (0.30 * (cols["h2o_scale"] * cols["rh"])[:, None] * (0.4 + g_h2o)
             + (0.55 / cols["vis_km"])[:, None] * (0.6 + 0.4 * slope)
             + 0.015 * (cols["p_hPa"] / 1013.25)[:, None]
             + 0.008 * cols["rainrt_mm_h"][:, None]
             + 0.05 * (cols["atmos_model"] == 3)[:, None] * (0.5 + g_gas)
             + 0.02 * (cols["icld"] == 18)[:, None])
    delta = k_ext * pf
    tau = np.exp(-delta)
    total = np.round(tau, 4)          # tape7 四位小数打印量化
    log_total = delta                  # LOG_TOTAL = -ln τ

    t_air = cols["t_ground_K"][:, None] - 8.0
    cos_sun = np.cos(np.radians(cols["sun_zenith_deg"]))[:, None]
    az_fac = 0.6 + 0.4 * np.cos(np.radians(cols["sun_rel_azimuth_deg"]))[:, None]
    haze_fac = (2.0 / cols["vis_km"])[:, None] ** 0.4

    if thermal:
        pth = planck(nu[None, :], t_air) * (1.0 - tau)
        sol = 4e-8 * cos_sun * az_fac * haze_fac * (0.3 + 0.2 * g_h2o) * np.sqrt(tau)
        lp = np.stack([tau, pth, sol, pth + sol], axis=1)   # TOT_TRANS PTH SOL TOTAL
        lp_cols = ["TOT_TRANS", "PTH_THRML", "SOL_SCAT", "TOTAL_RAD"]
        dv = 0.45 * (0.30 * (cols["h2o_scale"] * cols["rh"])[:, None] * (0.4 + g_h2o)
                     + (0.55 / cols["vis_km"])[:, None]) * 8.0
        tdv = np.exp(-dv)
        pthd = planck(nu[None, :], t_air - 12.0) * (1.0 - tdv)
        sold = 0.3 * 4e-8 * cos_sun * haze_fac * np.sqrt(tdv)
        ld = np.stack([tdv, pthd, sold, pthd + sold], axis=1)
    else:
        sol = 2e-6 * cos_sun * az_fac * haze_fac * (0.5 + 0.3 * slope) * tau ** 0.7
        sing = 0.55 * sol
        lp = np.stack([tau, sol, sing, 1.15 * sol], axis=1)
        lp_cols = ["TOT_TRANS", "SOL_SCAT", "SING_SCAT", "TOTAL_RAD"]
        ld = None

    # --------------------------------------------- 注入 failed / partial --
    status = np.zeros(N, dtype=np.uint8)
    n_fail = max(N // 50, 1)
    n_part = max(N // 33, 1)
    bad = rng.permutation(N)[: n_fail + n_part]
    fail_i, part_i = bad[:n_fail], bad[n_fail:]
    status[fail_i] = 2
    status[part_i] = 1
    total[fail_i] = 0.0
    log_total[fail_i] = 0.0
    lp[fail_i] = 0.0
    if ld is not None:
        ld[fail_i] = 0.0
    nan_i = part_i[: len(part_i) // 2]          # 一半 partial 带 NaN 列
    lp[nan_i, lp_cols.index("SOL_SCAT"), :] = np.nan

    # ------------------------------------------------------------- 落盘 --
    def save(name, arr, dtype):
        np.save(out / f"{name}.npy", np.ascontiguousarray(arr, dtype=dtype))

    save("params", params, np.float64)
    save("tau", np.stack([total, log_total], axis=1), np.float32)
    save("lpath", lp, np.float32)
    if ld is not None:
        save("ldown", ld, np.float32)
    save("wavenumber", nu, np.float32)
    save("index", np.arange(N), np.uint64)
    save("status", status, np.uint8)

    arrays = {
        "params": {"file": "params.npy", "dtype": "float64", "shape": [N, params.shape[1]]},
        "tau": {"file": "tau.npy", "dtype": "float32",
                "shape": [N, 2, K], "columns": TAU_COLS},
        "lpath": {"file": "lpath.npy", "dtype": "float32",
                  "shape": [N, 4, K], "columns": lp_cols},
        "wavenumber": {"file": "wavenumber.npy", "dtype": "float32",
                       "shape": [K], "units": "cm-1"},
        "index": {"file": "index.npy", "dtype": "uint64", "shape": [N]},
        "status": {"file": "status.npy", "dtype": "uint8", "shape": [N],
                   "legend": {"0": "ok", "1": "partial", "2": "failed"}},
    }
    if ld is not None:
        arrays["ldown"] = {"file": "ldown.npy", "dtype": "float32",
                           "shape": [N, 4, K], "columns": lp_cols}
    manifest = {
        "format": "ql-atmoforge-dataset-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": "make_fake_data.py (synthetic smoke-test data)",
        "seed": args.seed, "sampler": args.sampler,
        "n_samples": N, "n_present": N,
        "counts": {"ok": int((status == 0).sum()),
                   "partial": int((status == 1).sum()),
                   "failed": int((status == 2).sum())},
        "path_type": path_type, "band": band,
        "feature_names": feature_names, "sampled": spec,
        "fixed_effective": {"iday": 93, "ldown_zenith_deg": 45.0,
                            "o3_scale": 1.0, "co2_ppmv": 420.0},
        "arrays": arrays,
        "units": {"tau": "dimensionless transmittance (LOG_TOTAL is -ln)",
                  "lpath": "W cm-2 sr-1 / cm-1",
                  "ldown": "same as lpath"},
        "modtran": {"exe": "<synthetic>", "exe_size_bytes": 0,
                    "data_dir": "<synthetic>"},
        "earth_radius_km": EARTH_RADIUS_KM,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    n_sat = int((total.max(axis=1) < 1e-3).sum())
    print(f"[fake] {out}: N={N} K={K} thermal={thermal} path={path_type} "
          f"ok/partial/failed={manifest['counts']}  整谱饱和≈{n_sat}")


if __name__ == "__main__":
    main()
