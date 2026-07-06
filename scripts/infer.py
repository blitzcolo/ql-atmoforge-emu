#!/usr/bin/env python3
"""示例推理：从 run 目录加载代理网络，输出物理光谱（README §2.1）。

run 目录是自包含的（best.pt 含 EMA 权重 + norm/pca 副本），一次可加载多个
run（同一数据集的 tau/lpath/ldown 各一）。输入二选一：

  --data-dir <merge目录> --index N [N...]   取 params.npy 指定行；
                                            对应真值块存在时顺带打印物理误差
  --features-json f.json                    JSON 对象 {特征名: 值}，特征名与
                                            顺序以 manifest["feature_names"]
                                            为准（含派生几何列，横程 h2=h1、
                                            cos_view_zenith=cos(90°)≈0）；
                                            仍需 --data-dir 提供 manifest

输出 npz（--out）：
  wavenumber                    [K] cm-1
  params / feature_names        原始输入行及其列名
  <block>.<column>              [N, K] 物理量（辐亮度 W·cm-2·sr-1/cm-1）
  tau.delta / tau.tau           δ=-ln τ 与 τ=exp(-δ)（δ 在 0 处下钳）
  opaque                        [N] bool，δ_min > --opaque-delta 的视线；
                                部署契约：opaque 处 τ 置 0、像素=L_path
                                （τ 网在该区无监督，见 README §3）

例：
  python scripts/infer.py \\
      --run-dir runs/lwir_ground_v1_tau_256x4_muon \\
                runs/lwir_ground_v1_lpath_256x4_muon \\
                runs/lwir_ground_v1_ldown_256x4_muon \\
      --data-dir out/lwir_ground_v1_test --index 0 5 42 --out pred.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atmoemu.manifest import Manifest                 # noqa: E402
from atmoemu.model import build_model                 # noqa: E402
from atmoemu.transforms import InputSpec, OutputSpec  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", nargs="+", required=True,
                    help="一个或多个 run 目录（如 tau/lpath/ldown 各一）")
    ap.add_argument("--data-dir", required=True,
                    help="merge 数据集目录：提供 manifest（与 params/真值，若用 --index）")
    ap.add_argument("--index", type=int, nargs="*", default=None)
    ap.add_argument("--features-json", default=None)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--opaque-delta", type=float, default=7.0,
                    help="δ_min 超过此值的视线标记为不透明（τ<1e-3，"
                         "τ 网训练域之外）")
    ap.add_argument("--out", default=None, help="输出 npz 路径")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    return ap.parse_args()


def load_run(run_dir: str | Path, ckpt: str, device):
    """run 目录 -> (net 名, 前向闭包, OutputSpec, 指纹)。"""
    run = Path(run_dir)
    ck = torch.load(run / ckpt, map_location="cpu", weights_only=True)
    net, cfg = ck["net"], ck["config"]
    norm = json.loads((run / f"norm_{net}.json").read_text(encoding="utf-8"))
    ispec = InputSpec.from_json(norm["input"])
    ospec = OutputSpec.load(run / f"norm_{net}.targets.json",
                            run / f"norm_{net}.npz")
    pb = pm = None
    if cfg["pca_mode"] != "none":
        z = np.load(run / f"pca_{net}.npz")
        pb, pm = z["basis"], z["mean"]
    model = build_model(cfg, pca_basis=pb, pca_mean=pm)
    model.load_state_dict(ck["model_ema"])
    model.to(device).eval()

    # 辐亮度行的物理守卫：下界 0；log 通道上界 = 训练集通道最大值 × 2
    # （log_eps = 1e-4 × vmax，故 vmax 可从 norm 文件还原）。exp 逆变换
    # 无上界，暗场景里 z 稍偏就会在真值全死的深吸收带芯喷出比整谱真实
    # 能量高若干量级的假尖峰，摧毁带积分辐亮度——锚点过闸实测抓到 6 个
    # 量级的尖峰，封顶后消除。
    ceilings = {}
    for t, r in enumerate(ospec.rows):
        if r.kind != "delta" and r.log_mask is not None and r.log_mask.any():
            ceilings[t] = np.where(r.log_mask, r.log_eps * 1e4 * 2.0, np.inf)

    def forward(P: np.ndarray) -> np.ndarray:
        """原始参数行 [N, P] -> 物理量 [N, T, K]（辐亮度行已过物理守卫）。"""
        X = torch.from_numpy(ispec.apply(P)).to(device)
        with torch.no_grad():
            z = model(X).float()
            if cfg["pca_mode"] == "coeff":      # 系数头：用固定基解码回谱
                z = z @ torch.as_tensor(pb, dtype=torch.float32,
                                        device=device).T \
                    + torch.as_tensor(pm, dtype=torch.float32, device=device)
        Y = ospec.inverse(z.cpu().numpy())
        for t, r in enumerate(ospec.rows):
            if r.kind != "delta":
                Y[:, t] = np.maximum(Y[:, t], 0.0)
                if t in ceilings:
                    Y[:, t] = np.minimum(Y[:, t], ceilings[t])
        return Y

    return net, forward, ospec, ck["fingerprint"]


def assemble_params(man: Manifest, args) -> tuple[np.ndarray, np.ndarray | None]:
    """返回 (params 行 [N,P], 数据集行号或 None)。"""
    if (args.index is None) == (args.features_json is None):
        raise SystemExit("--index 与 --features-json 恰须给一个")
    if args.index is not None:
        idx = np.asarray(args.index, dtype=np.int64)
        P = np.asarray(man.npy("params", mmap=False))[idx]
        return P, idx
    feats = json.loads(Path(args.features_json).read_text(encoding="utf-8"))
    names = man.feature_names
    missing = [n for n in names if n not in feats]
    if missing:
        raise SystemExit(f"features-json 缺少特征 {missing}；"
                         f"完整布局（按序）= {names}")
    return np.asarray([[float(feats[n]) for n in names]]), None


def main():
    args = parse_args()
    man = Manifest.load(args.data_dir)
    P, idx = assemble_params(man, args)
    wn = man.wavenumber()
    print(f"[in] {len(P)} 行, 特征 {len(man.feature_names)} 维, "
          f"band={man.band['name']} K={man.K}")

    result: dict[str, np.ndarray] = {
        "wavenumber": wn.astype(np.float32),
        "params": P, "feature_names": np.asarray(man.feature_names),
    }
    opaque = None
    for rd in args.run_dir:
        net, forward, ospec, fp = load_run(rd, args.ckpt, args.device)
        man.check_compatible(fp)                # 特征布局/波段指纹必须一致
        Y = forward(P)                          # [N, T, K]
        for t, r in enumerate(ospec.rows):
            if r.kind == "delta":
                delta = np.maximum(Y[:, t], 0.0)      # δ 物理非负，下钳
                result["tau.delta"] = delta
                result["tau.tau"] = np.exp(-delta)
                opaque = delta.min(axis=1) > args.opaque_delta
            else:
                # 辐亮度物理非负；log(L+eps) 反变换在近零区可能给出 -eps 量级
                result[f"{r.block}.{r.column}"] = np.maximum(Y[:, t], 0.0)
            # 有真值就顺带报物理误差（来自 --index 模式的数据集块）
            if idx is not None and r.block in man.blocks:
                truth = np.asarray(man.npy(r.block)[idx, man.col_index(
                    r.block, r.column), :])
                if r.kind == "delta":
                    truth = np.minimum(truth, 20.0)   # 与训练目标同口径
                    err = np.abs(np.exp(-Y[:, t]) - np.exp(-truth))
                    print(f"  [{net}] {r.block}.{r.column}: "
                          f"τ MAE={err.mean():.4g} max={err.max():.4g}")
                else:
                    scale = float(np.abs(truth).max())
                    err = np.abs(Y[:, t] - truth)
                    pct = (f" (峰值的 {err.mean() / scale:.3%})"
                           if scale > 1e-30 else "（真值≈0，略去相对值）")
                    print(f"  [{net}] {r.block}.{r.column}: "
                          f"MAE={err.mean():.4g}{pct}")
    if opaque is not None:
        result["opaque"] = opaque
        if opaque.any():
            rows = np.nonzero(opaque)[0].tolist()
            print(f"[gate] {int(opaque.sum())}/{len(P)} 行 δ_min > "
                  f"{args.opaque_delta}（不透明，部署时 τ 置 0）: 行 {rows}")

    if args.out:
        np.savez_compressed(args.out, **result)
        print(f"[done] -> {args.out}  keys={sorted(result)}")


if __name__ == "__main__":
    main()
