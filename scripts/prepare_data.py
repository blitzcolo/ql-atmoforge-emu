#!/usr/bin/env python3
"""数据清洗 + 归一化统计 + 训练/验证/测试划分（ModModel.md §4.2 训练前过滤、§5、§6）。

对一个 `ql-atmoforge merge` 产出目录：
  1. 过滤 status=2（全零占位）、可选剔除 partial、NaN 目标列、整谱饱和样本；
  2. 自动划分 train/val(/test)。首选 --test-dir 指向独立 random 采样数据集
     （Sobol 序列任何子段互不独立，同集切分只可用于冒烟）；
  3. 逐网（tau/lpath/ldown）生成输入解析归一化 spec + 输出 μ/σ 统计（方差下限）；
  4. 可选拟合 PCA 基（vis 必须；--pca lpath=100）。

产物写入 <data-dir>/prep/：
  keep_mask.npy  train_idx.npy  val_idx.npy  [test_idx.npy]  splits.json
  norm_<net>.json  norm_<net>.targets.json  norm_<net>.npz  [pca_<net>.npz]

用法：
  python scripts/prepare_data.py --data-dir out/lwir_ground_v1 \
      --test-dir out/lwir_ground_v1_test
  python scripts/prepare_data.py --data-dir out/vis_ground_v1 \
      --pca lpath=100 tau=150
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atmoemu.manifest import Manifest, STATUS_FAILED, STATUS_OK  # noqa: E402
from atmoemu.transforms import (build_input_spec, default_targets,  # noqa: E402
                                fit_output_spec)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True, help="merge 产出目录")
    ap.add_argument("--nets", nargs="+", default=None,
                    choices=["tau", "lpath", "ldown"],
                    help="默认 tau lpath（thermal 波段自动加 ldown）")
    ap.add_argument("--test-dir", default=None,
                    help="独立 random 采样测试集目录（推荐，做兼容性校验后记录）")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.0,
                    help="仅在无 --test-dir 时使用；同集切分带 Sobol 相关性，慎用")
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--drop-partial", action="store_true",
                    help="剔除 partial 样本（默认保留，仅剔目标列含 NaN 的）")
    ap.add_argument("--saturation-tau-max", type=float, default=1e-3,
                    help="max_ν(tau.TOTAL) 低于此阈值的整谱饱和样本剔除；0 关闭")
    ap.add_argument("--delta-clamp", type=float, default=20.0)
    ap.add_argument("--log-dynrange", type=float, default=1e3,
                    help="辐亮度通道动态范围超过此值时改 log(L+ε) 训练")
    ap.add_argument("--var-floor-frac", type=float, default=0.01)
    ap.add_argument("--pca", nargs="*", default=[], metavar="NET=NCOMP",
                    help="按网拟合 PCA 基，如 --pca lpath=100 tau=150")
    ap.add_argument("--pca-max-samples", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=2048)
    return ap.parse_args()


def scan_bad_rows(man: Manifest, needed: dict[str, list[int]],
                  candidates: np.ndarray, chunk: int) -> np.ndarray:
    """candidates 中目标列含 NaN 的行（mmap 分块）。needed: block -> col indices。"""
    bad = np.zeros(len(candidates), dtype=bool)
    for block, cols in needed.items():
        arr = man.npy(block)
        for s in range(0, len(candidates), chunk):
            idx = candidates[s:s + chunk]
            a = arr[idx][:, cols, :]
            bad[s:s + len(idx)] |= ~np.isfinite(a).all(axis=(1, 2))
    return bad


def scan_saturated(man: Manifest, candidates: np.ndarray,
                   thr: float, chunk: int) -> np.ndarray:
    cols = man.block_columns("tau")
    if "TOTAL" not in cols:
        print("[warn] tau 块无 TOTAL 列，跳过饱和过滤")
        return np.zeros(len(candidates), dtype=bool)
    c = cols.index("TOTAL")
    arr = man.npy("tau")
    sat = np.zeros(len(candidates), dtype=bool)
    for s in range(0, len(candidates), chunk):
        idx = candidates[s:s + chunk]
        mx = np.asarray(arr[idx, c, :]).max(axis=1)
        sat[s:s + len(idx)] = mx < thr
    return sat


def fit_pca(man: Manifest, ospec, train_idx: np.ndarray, n_comp: int,
            max_samples: int, chunk: int, out_path: Path) -> None:
    import torch
    sub = train_idx[:max_samples]
    d = ospec.d_out
    Y = np.empty((len(sub), d), dtype=np.float32)
    arrs = {r.block: man.npy(r.block) for r in ospec.rows}
    for s in range(0, len(sub), chunk):
        idx = sub[s:s + chunk]
        raw = np.stack([np.asarray(arrs[r.block][idx, r.col_index, :])
                        for r in ospec.rows], axis=1)
        Y[s:s + len(idx)] = ospec.transform(raw)
    t = torch.from_numpy(Y)
    mean = t.mean(dim=0)
    tc = t - mean
    q = min(n_comp + 16, min(tc.shape))
    _, S, V = torch.svd_lowrank(tc, q=q, niter=4)
    V = V[:, :n_comp].contiguous()                      # [d_out, n_comp]
    total = float((tc ** 2).sum())
    kept = float((S[:n_comp] ** 2).sum())
    resid = tc - (tc @ V) @ V.T
    recon_nrmse = float(torch.sqrt((resid ** 2).mean()))
    np.savez_compressed(out_path, basis=V.numpy(), mean=mean.numpy(),
                        explained=kept / max(total, 1e-30),
                        recon_nrmse_std=recon_nrmse, n_fit=len(sub))
    print(f"  PCA n_comp={n_comp}  方差保留 {kept / max(total, 1e-30):.5f}"
          f"  重构 NRMSE(标准化空间) {recon_nrmse:.4g} -> {out_path.name}")


def main():
    args = parse_args()
    man = Manifest.load(args.data_dir)
    prep = Path(args.data_dir) / "prep"
    prep.mkdir(exist_ok=True)

    nets = args.nets or (["tau", "lpath", "ldown"] if man.band["thermal"]
                         else ["tau", "lpath"])
    print(f"[dataset] {man.data_dir.name}: band={man.band['name']} "
          f"K={man.K} path={man.path_type} sampler={man.sampler} "
          f"N={man.n_present}  nets={nets}")

    # ------------------------------------------------------------- 清洗 --
    status = np.asarray(man.npy("status", mmap=False))
    N = len(status)
    keep = status != STATUS_FAILED
    n_failed = int((~keep).sum())
    n_partial = int((status == 1).sum())
    if args.drop_partial:
        keep &= status == STATUS_OK

    cand = np.where(keep)[0]
    if args.saturation_tau_max > 0 and "tau" in man.blocks:
        sat = scan_saturated(man, cand, args.saturation_tau_max, args.chunk)
        keep[cand[sat]] = False
        print(f"[clean] 整谱饱和 (max τ < {args.saturation_tau_max:g}): "
              f"剔 {int(sat.sum())}")
        cand = cand[~sat]

    needed: dict[str, list[int]] = {}
    for net in nets:
        for block, column, _ in default_targets(man, net):
            needed.setdefault(block, [])
            ci = man.col_index(block, column)
            if ci not in needed[block]:
                needed[block].append(ci)
    bad = scan_bad_rows(man, needed, cand, args.chunk)
    keep[cand[bad]] = False
    print(f"[clean] status=failed 剔 {n_failed}，partial "
          f"{'剔' if args.drop_partial else '保留'} {n_partial}，"
          f"目标列 NaN 剔 {int(bad.sum())}，最终保留 {int(keep.sum())}/{N}")

    # ------------------------------------------------------------- 划分 --
    kept_idx = np.where(keep)[0]
    rng = np.random.default_rng(args.split_seed)
    perm = kept_idx[rng.permutation(len(kept_idx))]
    n_val = int(round(args.val_frac * len(perm)))
    test_frac = args.test_frac
    if args.test_dir:
        Manifest.load(args.test_dir).check_compatible(man.fingerprint())
        test_frac = 0.0
        print(f"[split] 测试集 = 独立数据集 {args.test_dir}（已校验兼容）")
    elif man.sampler == "sobol":
        print("[split][警告] 无 --test-dir：Sobol 序列子段互不独立，"
              "同集切分的 val/test 只可用于冒烟与相对比较，"
              "正式评估请生成 sampler=random 的独立测试集")
    n_test = int(round(test_frac * len(perm)))
    val_idx = np.sort(perm[:n_val])
    test_idx = np.sort(perm[n_val:n_val + n_test])
    train_idx = np.sort(perm[n_val + n_test:])
    print(f"[split] train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx) if n_test else f'外部({args.test_dir})'}")

    np.save(prep / "keep_mask.npy", keep)
    np.save(prep / "train_idx.npy", train_idx)
    np.save(prep / "val_idx.npy", val_idx)
    if n_test:
        np.save(prep / "test_idx.npy", test_idx)

    # ------------------------------------------------- 逐网归一化 + PCA --
    pca_req = {}
    for kv in args.pca:
        k, v = kv.split("=")
        pca_req[k] = int(v)

    for net in nets:
        print(f"[net={net}]")
        ispec = build_input_spec(man, net)
        # 抽样健壮性检查：变换后有限、无常量列
        probe = ispec.apply(np.asarray(man.npy("params", mmap=False))[train_idx[:2048]])
        assert np.isfinite(probe).all(), f"net={net}: 输入变换出现非有限值"
        dead = [n for n, s in zip(ispec.feature_names_out, probe.std(axis=0))
                if s < 1e-7]
        if dead:
            print(f"  [warn] 近常量输入列（检查配置范围）: {dead}")
        print(f"  输入 {ispec.d_in} 维: {ispec.feature_names_out}")

        targets = default_targets(man, net)
        ospec = fit_output_spec(
            man, targets, train_idx, delta_clamp=args.delta_clamp,
            log_dynrange=args.log_dynrange, floor_frac=args.var_floor_frac,
            chunk=args.chunk)
        (prep / f"norm_{net}.json").write_text(
            json.dumps({"net": net, "input": ispec.to_json(),
                        "targets": [list(t) for t in targets]},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        ospec.save(prep / f"norm_{net}.targets.json", prep / f"norm_{net}.npz")

        if net in pca_req:
            fit_pca(man, ospec, train_idx, pca_req[net],
                    args.pca_max_samples, args.chunk, prep / f"pca_{net}.npz")

    (prep / "splits.json").write_text(json.dumps({
        "data_dir": str(Path(args.data_dir).resolve()),
        "test_dir": str(Path(args.test_dir).resolve()) if args.test_dir else None,
        "val_frac": args.val_frac, "test_frac": test_frac,
        "split_seed": args.split_seed, "drop_partial": args.drop_partial,
        "saturation_tau_max": args.saturation_tau_max,
        "delta_clamp": args.delta_clamp,
        "n_total": int(N), "n_keep": int(keep.sum()),
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": int(n_test),
        "nets": nets, "fingerprint": man.fingerprint(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] 产物写入 {prep}")


if __name__ == "__main__":
    main()
