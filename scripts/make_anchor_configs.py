#!/usr/bin/env python3
"""锚点配置生成（ModModel.md §4.2 第 4 步 / §7.3）。

把 atmospheres.json 的 8 个场景预设映射成每个生产数据集口径下的单样本
锚点 config（8 预设 × 10 数据集 = 80 份），直跑 MODTRAN 得真值，供
evaluate.py 对网络逐一过闸。

布局保持规则（evaluate 的指纹校验只认 band/path_type/feature_names）：
  - `sampled` 的键与顺序保持与生产配置完全一致；
  - 每个采样维退化成单点 `{"values": [v]}`（config.cpp 拒绝 lo==hi 的
    uniform，而 values 不限参数类型）；
  - `fixed` 原样保留——被生产口径钉死的参数（如 vis 波段的 t_ground）
    不吸收预设值，锚点是"该数据集口径下"的点，不是预设的完整复刻。

预设 → 参数映射（超出生产采样域的值钳制到边界并警告）：
  visibility_km→vis_km  humidity→rh  temperature_k→t_ground_K
  pressure_hPa→p_hPa  precipitation_rate_mm_h→rainrt_mm_h
  modtran_model→atmos_model（1 不在 {2,3} 训练域 → 收到 2）
  aerosol: Rural→1 Maritime→4 Urban→5 Industrial→5 Desert→10；
           weather==Fog 强制 ihaze=9（辐射雾）
  icld: 降雨>0 → 6（雨模型），否则 0（雾由 ihaze 表达，不叠云）
  sun_zenith_deg→原值；sun_azimuth_deg→±Δφ 镜像折入 [0,180]
  几何取域内代表点：横程 h1=0.1 km、range=1 km；斜程 h1=3 km、θ=135°
  h2o_scale=1.0

用法：
  python scripts/make_anchor_configs.py --forge-root /mnt/d/ql-atmoforge
  产物：<forge-root>/configs/anchors/<dataset>__<preset>.json
        out_dir = out/anchors/<dataset>/<preset>（相对 gen 时的 cwd）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = [f"{b}_{p}_v1" for b in ("lwir", "mwir", "nir", "swir", "vis")
            for p in ("ground", "slant")]
SLUGS = ["01_clear", "02_turb_clear", "03_urban_haze", "04_fog",
         "05_light_rain", "06_heavy_rain", "07_snow", "08_haze"]
AEROSOL_IHAZE = {"Rural": 1, "Maritime": 4, "Urban": 5,
                 "Industrial": 5, "Desert": 10}
GEOM = {  # 域内代表几何
    "horizontal": {"h1_km": 0.1, "range_km": 1.0},
    "slant_to_ground": {"h1_km": 3.0, "view_zenith_deg": 135.0},
}


def dist_bounds(dist):
    if "values" in dist:
        return min(dist["values"]), max(dist["values"])
    for k in ("uniform", "log_uniform"):
        if k in dist:
            return dist[k][0], dist[k][1]
    raise ValueError(f"unknown dist: {dist}")


def preset_value(name: str, pre: dict, path_type: str):
    """采样维名 → 预设映射值；None = 无映射（用生产范围中点）。"""
    if name == "atmos_model":
        return pre["modtran_model"]
    if name == "ihaze":
        if pre["weather"] == "Fog":
            return 9
        return AEROSOL_IHAZE.get(pre["aerosol"])
    if name == "icld":
        return 6 if pre["precipitation_rate_mm_h"] > 0 else 0
    if name == "vis_km":
        return pre["visibility_km"]
    if name == "rainrt_mm_h":
        return pre["precipitation_rate_mm_h"]
    if name == "t_ground_K":
        return pre["temperature_k"]
    if name == "rh":
        return pre["humidity"]
    if name == "p_hPa":
        return pre["pressure_hPa"]
    if name == "h2o_scale":
        return 1.0
    if name == "sun_zenith_deg":
        return pre["sun_zenith_deg"]
    if name == "sun_rel_azimuth_deg":
        a = abs(pre["sun_azimuth_deg"]) % 360.0
        return 360.0 - a if a > 180.0 else a      # ±Δφ 镜像对称
    if name in GEOM[path_type]:
        return GEOM[path_type][name]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forge-root", default="/mnt/d/ql-atmoforge")
    ap.add_argument("--out-prefix", default="out/anchors",
                    help="锚点数据集 out_dir 前缀（gen 时相对 cwd）")
    args = ap.parse_args()
    root = Path(args.forge_root)
    presets = json.loads((root / "configs/atmospheres.json")
                         .read_text(encoding="utf-8"))["atmospheres"]
    assert len(presets) == len(SLUGS), "预设数与 SLUGS 不符"
    adir = root / "configs/anchors"
    adir.mkdir(exist_ok=True)

    n_written = 0
    for ds in DATASETS:
        src = root / "configs" / f"{ds}.json"
        if not src.exists():
            print(f"[skip] {src} 不存在")
            continue
        cfg0 = json.loads(src.read_text(encoding="utf-8"))
        for pre, slug in zip(presets, SLUGS):
            cfg = json.loads(json.dumps(cfg0))       # deep copy
            cfg["run"].update({
                "out_dir": f"{args.out_prefix}/{ds}/{slug}",
                "n_samples": 1, "workers": 1,
                "sampler": "random", "csv_preview": 0,
            })
            for name, dist in cfg["sampled"].items():
                v = preset_value(name, pre, cfg["path_type"])
                lo, hi = dist_bounds(dist)
                if v is None:
                    v = 0.5 * (lo + hi)
                if "values" in dist:                  # 离散：收到最近的允许值
                    vals = dist["values"]
                    v2 = min(vals, key=lambda x: abs(x - v))
                    if v2 != v:
                        print(f"[clamp] {ds}/{slug} {name}: {v} -> {v2}"
                              f"（训练域 {vals}）")
                    v = v2
                else:                                  # 连续：钳到生产范围
                    v2 = min(max(v, lo), hi)
                    if v2 != v:
                        print(f"[clamp] {ds}/{slug} {name}: {v} -> {v2}"
                              f"（训练域 [{lo}, {hi}]）")
                    v = v2
                cfg["sampled"][name] = {"values": [v]}
            out = adir / f"{ds}__{slug}.json"
            out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
            n_written += 1
    print(f"[done] {n_written} 份锚点配置 -> {adir}")


if __name__ == "__main__":
    main()
