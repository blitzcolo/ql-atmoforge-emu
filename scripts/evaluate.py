#!/usr/bin/env python3
"""在独立测试集 / 锚点集上评估训练好的代理网络（ModModel.md §7.3）。

  python scripts/evaluate.py --run-dir runs/lwir_ground_v1_tau_256x4_adamw \
      --data-dir out/lwir_ground_v1_test

--data-dir 可以是任何与训练集同波段、同特征布局的 merge 产出目录
（独立 random 测试集、atmospheres.json 锚点集直跑结果等）。
若指向训练用数据集本身，则自动使用 prep/ 里的 test_idx（无则 val_idx，并告警）。

输出: <run-dir>/eval_<dataset>.json（标量指标）+ eval_<dataset>.channels.npz。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atmoemu.manifest import Manifest, STATUS_FAILED        # noqa: E402
from atmoemu.metrics import compute_metrics, format_metrics  # noqa: E402
from atmoemu.model import build_model                       # noqa: E402
from atmoemu.transforms import InputSpec, OutputSpec        # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--use-raw-weights", action="store_true",
                    help="用原始权重而非 EMA 权重")
    ap.add_argument("--saturation-tau-max", type=float, default=1e-3,
                    help="外部数据集清洗用（与训练侧保持一致）")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = 全部")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=2048)
    return ap.parse_args()


def clean_indices(man: Manifest, ospec: OutputSpec, sat_thr: float,
                  chunk: int) -> np.ndarray:
    status = np.asarray(man.npy("status", mmap=False))
    keep = status != STATUS_FAILED
    cand = np.where(keep)[0]
    if sat_thr > 0 and "tau" in man.blocks and "TOTAL" in man.block_columns("tau"):
        c = man.block_columns("tau").index("TOTAL")
        arr = man.npy("tau")
        drop = np.zeros(len(cand), dtype=bool)
        for s in range(0, len(cand), chunk):
            idx = cand[s:s + chunk]
            drop[s:s + len(idx)] = np.asarray(arr[idx, c, :]).max(axis=1) < sat_thr
        cand = cand[~drop]
    bad = np.zeros(len(cand), dtype=bool)
    for r in ospec.rows:
        arr = man.npy(r.block)
        for s in range(0, len(cand), chunk):
            idx = cand[s:s + chunk]
            a = np.asarray(arr[idx, r.col_index, :])
            bad[s:s + len(idx)] |= ~np.isfinite(a).all(axis=1)
    return cand[~bad]


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(run_dir / args.ckpt, map_location=device, weights_only=True)
    net = ck["net"]
    model_cfg = ck["config"]

    norm = json.loads((run_dir / f"norm_{net}.json").read_text(encoding="utf-8"))
    ispec = InputSpec.from_json(norm["input"])
    ospec = OutputSpec.load(run_dir / f"norm_{net}.targets.json",
                            run_dir / f"norm_{net}.npz")
    pca_basis = pca_mean = None
    if model_cfg["pca_mode"] != "none":
        z = np.load(run_dir / f"pca_{net}.npz")
        pca_basis, pca_mean = z["basis"], z["mean"]

    man = Manifest.load(args.data_dir)
    man.check_compatible(ck["fingerprint"])

    # ------------------------------------------------------- eval indices --
    prep = man.data_dir / "prep"
    splits_f = prep / "splits.json"
    if splits_f.exists() and json.loads(splits_f.read_text(encoding="utf-8"))[
            "data_dir"] == str(man.data_dir.resolve()):
        if (prep / "test_idx.npy").exists():
            idx, src = np.load(prep / "test_idx.npy"), "prep/test_idx"
        else:
            idx, src = np.load(prep / "val_idx.npy"), "prep/val_idx"
            print("[warn] 评估集 = 训练数据集的 val 切分（与训练集同 Sobol 序列，"
                  "结果偏乐观），请优先用独立 random 测试集")
    else:
        idx, src = clean_indices(man, ospec, args.saturation_tau_max,
                                 args.chunk), "external-clean"
    if args.max_samples > 0:
        idx = idx[:args.max_samples]
    print(f"[eval] {man.data_dir.name} ({src}, sampler={man.sampler}): "
          f"{len(idx)} 样本, net={net}, device={device}")

    # ------------------------------------------------------------ forward --
    model = build_model(model_cfg, pca_basis=pca_basis, pca_mean=pca_mean)
    model.load_state_dict(ck["model_raw" if args.use_raw_weights else "model_ema"])
    model.to(device).eval()

    params = np.asarray(man.npy("params", mmap=False))
    X = torch.from_numpy(ispec.apply(params[idx]))
    arrs = {r.block: man.npy(r.block) for r in ospec.rows}
    z_true = np.empty((len(idx), ospec.d_out), dtype=np.float32)
    for s in range(0, len(idx), args.chunk):
        ii = idx[s:s + args.chunk]
        raw = np.stack([np.asarray(arrs[r.block][ii, r.col_index, :])
                        for r in ospec.rows], axis=1)
        z_true[s:s + len(ii)] = ospec.transform(raw)

    v = None
    if model_cfg["pca_mode"] == "coeff":
        v = (torch.as_tensor(pca_basis, dtype=torch.float32, device=device),
             torch.as_tensor(pca_mean, dtype=torch.float32, device=device))
    z_pred = np.empty_like(z_true)
    with torch.no_grad():
        for s in range(0, len(X), args.batch):
            p = model(X[s:s + args.batch].to(device)).float()
            if v is not None:
                p = p @ v[0].T + v[1]
            z_pred[s:s + len(p)] = p.cpu().numpy()

    # ------------------------------------------------------------ metrics --
    m = compute_metrics(z_true, z_pred, ospec, man.wavenumber(),
                        thermal=bool(man.band["thermal"]))
    print(format_metrics(m))
    per_ch = m.pop("per_channel_nrmse")
    report = {"data_dir": str(man.data_dir.resolve()), "source": src,
              "n_samples": int(len(idx)), "net": net,
              "ckpt": args.ckpt, "ema": not args.use_raw_weights,
              "metrics": m}
    out = run_dir / f"eval_{man.data_dir.name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    np.savez_compressed(run_dir / f"eval_{man.data_dir.name}.channels.npz",
                        per_channel_nrmse=per_ch,
                        wavenumber=man.wavenumber())
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
