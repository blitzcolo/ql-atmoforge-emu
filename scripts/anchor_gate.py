#!/usr/bin/env python3
"""锚点过闸（ModModel.md §7.3 最后一道验收）。

对每张网 × 每个锚点数据集（make_anchor_configs.py + gen/merge 产物）做
"网络 vs 直跑 MODTRAN 真值"的端到端校验。锚点是 n=1 的单场景点，
population 型指标（evaluate.py 的逐通道尺度）在 n=1 退化，这里用
单谱口径原生计算：

  tau 网      τ_mae = mean|exp(−δ̂) − exp(−δ)|；真值 max τ < 1e-3 记 SAT
              （整谱饱和场景 τ≡0，τ 网无监督也无需监督）
  辐亮度网    有效元素 = L_true > 1% × 该谱峰值；
              热波段判 bt_mae_K（逆 Planck 均值），反射波段判 rel 的
              **中位数**——单谱下均值会被个别带缘通道主导，中位反映
              典型通道精度；深吸收带芯假尖峰对带积分的威胁已由
              infer.load_run 前向守卫（log 通道 2×训练最大值封顶）兜底；
              多目标网（mwir lpath）逐目标判，取最差；
              暗场景（谱峰 < 1e-6 × 该数据集 8 锚点的真值峰值，如浓雾里
              的太阳散射——真值和预测都是数值零）该目标行跳过

用法：
  python scripts/anchor_gate.py --anchors-root /mnt/d/ql-atmoforge/out/anchors \
      [--runs runs] [--only lwir_ground_v1] [--tau-thr 0.02] [--bt-thr 1.0] \
      [--rel-thr 0.05]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atmoemu.manifest import Manifest            # noqa: E402
from atmoemu.metrics import brightness_temperature  # noqa: E402
from infer import load_run                       # noqa: E402

SLUGS = ["01_clear", "02_turb_clear", "03_urban_haze", "04_fog",
         "05_light_rain", "06_heavy_rain", "07_snow", "08_haze"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--anchors-root", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tau-thr", type=float, default=0.02)
    ap.add_argument("--bt-thr", type=float, default=1.0)
    ap.add_argument("--rel-thr", type=float, default=0.05)
    return ap.parse_args()


def eval_anchor(net, fwd, ospec, man, thermal, args, band_scale):
    """→ (显示值, verdict)。verdict: ok / FAIL / SAT。
    band_scale: {row_idx: 该数据集全部锚点的真值峰值}，暗场景判据用。"""
    P = np.asarray(man.npy("params", mmap=False))[:1]
    Y = fwd(P)                                     # [1, T, K]
    nu = man.wavenumber()
    worst_v, worst_bad = -np.inf, False
    any_live = False
    for t, r in enumerate(ospec.rows):
        a = np.asarray(man.npy(r.block)[0, man.col_index(r.block, r.column), :],
                       dtype=np.float64)
        b = Y[0, t]
        if r.kind == "delta":
            a = np.minimum(a, r.clamp_max or 20.0)
            if np.exp(-a).max() < 1e-3:
                continue                           # 饱和场景：τ 网域外
            any_live = True
            v = float(np.abs(np.exp(-np.maximum(b, 0.0)) - np.exp(-a)).mean())
            bad = v > args.tau_thr
        else:
            if a.max() < 1e-6 * band_scale.get(t, 0.0):
                continue                           # 暗场景：无信号可锚定
            mask = a > 1e-2 * a.max()
            if not mask.any():
                continue
            any_live = True
            bp = np.maximum(b, 0.0)
            if thermal and r.column in ("TOTAL_RAD", "PTH_THRML"):
                dt = np.abs(brightness_temperature(np.where(mask, bp, np.nan), nu)
                            - brightness_temperature(np.where(mask, a, np.nan), nu))
                v = float(np.nanmean(dt))
                bad = v > args.bt_thr
            else:
                v = float(np.median(np.abs(bp - a)[mask] / a[mask]))
                bad = v > args.rel_thr
        if v > worst_v:
            worst_v, worst_bad = v, bad
    if not any_live:
        return float("nan"), "SAT"
    return worst_v, "FAIL" if worst_bad else "ok"


def main():
    args = parse_args()
    root = Path(args.anchors_root)
    rows_out = []
    fails = sats = 0
    for run in sorted(Path(args.runs).iterdir()):
        if not (run / "best.pt").exists():
            continue
        m = re.match(r"(\w+?_v1)_(tau|lpath|ldown)_", run.name)
        if not m:
            continue
        ds, net = m.group(1), m.group(2)
        if args.only and ds not in args.only:
            continue
        thermal = ds.startswith(("lwir", "mwir"))
        loaded = None
        cells = []
        band_scale: dict[int, float] = {}
        for slug in SLUGS:                          # 预扫：各目标行的波段量级
            adir = root / ds / slug
            if not (adir / "manifest.json").exists():
                continue
            if loaded is None:
                loaded = load_run(run, "best.pt", "cpu")
            man = Manifest.load(adir)
            for t, r in enumerate(loaded[2].rows):
                if r.kind == "delta":
                    continue
                a = np.asarray(man.npy(r.block)[0, man.col_index(
                    r.block, r.column), :], dtype=np.float64)
                band_scale[t] = max(band_scale.get(t, 0.0), float(a.max()))
        for slug in SLUGS:
            adir = root / ds / slug
            if not (adir / "manifest.json").exists():
                cells.append(f"{'MISSING':>8s}")
                continue
            _, fwd, ospec, fp = loaded
            man = Manifest.load(adir)
            man.check_compatible(fp)
            v, verdict = eval_anchor(net, fwd, ospec, man, thermal, args,
                                     band_scale)
            if verdict == "ok":
                cells.append(f"{v:8.3g}")
            elif verdict == "SAT":
                cells.append(f"{'SAT':>8s}"); sats += 1
            else:
                cells.append(f"{('!'+format(v,'.3g')):>8s}"); fails += 1
        rows_out.append((run.name, cells))

    print(f"\n{'run':44s} " + " ".join(f"{s.split('_', 1)[0]:>8s}" for s in SLUGS))
    for name, cells in rows_out:
        print(f"{name:44s} " + " ".join(cells))
    print(f"\n判据: tau_mae<={args.tau_thr} | bt_mae_K<={args.bt_thr}"
          f" | rad_rel_mae<={args.rel_thr}；!值 = 超阈；"
          f"SAT = 整谱饱和跳过\nFAIL={fails}  SAT={sats}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
