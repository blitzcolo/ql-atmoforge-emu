#!/usr/bin/env python3
"""训练一个代理网络（tau / lpath / ldown），单卡或 torchrun 多卡 DDP。

先运行 scripts/prepare_data.py 生成 prep/ 产物。

单卡:
  python scripts/train.py --data-dir out/lwir_ground_v1 --net tau
多卡 (以 4 卡为例):
  torchrun --standalone --nproc_per_node=4 scripts/train.py \
      --data-dir out/mwir_slant_v1 --net lpath --batch-size 1024
vis + PCA 头 (prepare 时需 --pca lpath=100):
  python scripts/train.py --data-dir out/vis_ground_v1 --net lpath \
      --pca-mode head --width 512 --blocks 6

配置来源优先级: 命令行 > --config JSON > 默认值。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atmoemu.data import SpectraDataset, load_prep          # noqa: E402
from atmoemu.model import EMA, ResMLP, build_model          # noqa: E402
from atmoemu.optim import build_optimizer                   # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None, help="JSON 配置文件（键 = 参数名）")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--net", required=True, choices=["tau", "lpath", "ldown"])
    ap.add_argument("--prep-dir", default=None, help="默认 <data-dir>/prep")
    ap.add_argument("--out", default=None, help="默认 runs/<dataset>_<net>_<tag>")
    # 结构（ModModel.md §7.1: width 256–512, blocks 4–6）
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--pca-mode", default="none", choices=["none", "head", "coeff"])
    # 训练（§7.2）
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1024, help="每卡 batch")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--min-lr-frac", type=float, default=0.01)
    ap.add_argument("--huber-delta", type=float, default=0.0,
                    help=">0 时用 Huber 损失（离群多时）")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--patience", type=int, default=30,
                    help="验证损失连续多少个 epoch 无改善则早停")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-bf16", action="store_true", help="关闭 bf16 混合精度")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--preload", action="store_true",
                    help="目标谱整体载入内存（mwir/lwir 建议开）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="last.pt 路径")
    ap.add_argument("--export-safetensors", default=None, metavar="OUT_DIR",
                    help="训练结束后把 best.pt 导出为 safetensors 包")
    return merge_config(ap)


def merge_config(ap: argparse.ArgumentParser):
    args, _ = ap.parse_known_args()
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    return ap.parse_args()


def ddp_env():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def lr_scale(step: int, total: int, warmup_frac: float, min_frac: float) -> float:
    t = step / max(total, 1)
    if t < warmup_frac:
        return t / max(warmup_frac, 1e-9)
    p = (t - warmup_frac) / max(1.0 - warmup_frac, 1e-9)
    return min_frac + 0.5 * (1.0 - min_frac) * (1.0 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def validate(model_cfg, ema, val_X, val_Y, device, pca, batch=8192):
    """EMA 权重、标准化谱空间的验证损失（coeff 模式先解码，含截断残差）。"""
    m = build_model(model_cfg,
                    pca_basis=None if pca is None else pca["basis"],
                    pca_mean=None if pca is None else pca["mean"]).to(device)
    m.load_state_dict(ema.shadow_on(m))
    m.eval()
    v = vm = None
    if model_cfg["pca_mode"] == "coeff":
        v = torch.as_tensor(pca["basis"], dtype=torch.float32, device=device)
        vm = torch.as_tensor(pca["mean"], dtype=torch.float32, device=device)
    se, n = 0.0, 0
    for s in range(0, len(val_X), batch):
        x = val_X[s:s + batch].to(device)
        y = val_Y[s:s + batch].to(device)
        p = m(x).float()
        if v is not None:
            p = p @ v.T + vm
        se += F.mse_loss(p, y, reduction="sum").item()
        n += y.numel()
    return se / max(n, 1)


def main():
    args = parse_args()
    rank, world, local = ddp_env()
    is_main = rank == 0
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and not args.no_bf16
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    man, ispec, ospec, splits, pca_npz = load_prep(
        args.data_dir, args.net, args.prep_dir)
    pca = None
    if args.pca_mode != "none":
        if pca_npz is None:
            raise SystemExit(f"--pca-mode {args.pca_mode} 需要先在 prepare_data.py "
                             f"用 --pca {args.net}=<n> 拟合 PCA 基")
        pca = {"basis": pca_npz["basis"], "mean": pca_npz["mean"]}

    train_idx = splits["idx"]["train"]
    val_idx = splits["idx"]["val"]
    train_ds = SpectraDataset(man, train_idx, ispec, ospec, preload=args.preload)
    val_ds = SpectraDataset(man, val_idx, ispec, ospec, preload=True)
    # 验证集常驻内存张量（rank0 用）
    val_X = val_ds.X
    val_Y = val_ds._preloaded

    model_cfg = {"d_in": ispec.d_in, "d_out": ospec.d_out,
                 "width": args.width, "blocks": args.blocks,
                 "pca_mode": args.pca_mode}
    model = build_model(model_cfg,
                        pca_basis=None if pca is None else pca["basis"],
                        pca_mean=None if pca is None else pca["mean"]).to(device)
    if is_main:
        print(f"[model] d_in={ispec.d_in} d_out={ospec.d_out} "
              f"width={args.width} blocks={args.blocks} pca={args.pca_mode} "
              f"params={model.n_params / 1e6:.2f}M  device={device} "
              f"bf16={use_bf16} world={world}")

    raw_model = model
    if world > 1:
        model = DDP(model, device_ids=[local] if device.type == "cuda" else None)

    opt = build_optimizer(raw_model, args.optimizer, args.lr, args.weight_decay)
    ema = EMA(raw_model, decay=args.ema_decay)
    ema.shadow_on = lambda m: {k: v.to(next(m.parameters()).device)
                               for k, v in ema.state_dict_for(m).items()}

    sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                  shuffle=True, drop_last=True)
               if world > 1 else None)
    loader = DataLoader(train_ds, batch_size=args.batch_size,
                        shuffle=sampler is None, sampler=sampler,
                        num_workers=args.num_workers,
                        pin_memory=device.type == "cuda",
                        persistent_workers=args.num_workers > 0,
                        drop_last=True)
    steps_per_epoch = max(len(loader), 1)
    total_steps = args.epochs * steps_per_epoch

    v_t = vm_t = None
    if args.pca_mode == "coeff":
        v_t = torch.as_tensor(pca["basis"], dtype=torch.float32, device=device)
        vm_t = torch.as_tensor(pca["mean"], dtype=torch.float32, device=device)

    # ---------------------------------------------------------- run dir --
    ds_name = Path(args.data_dir).resolve().name
    tag = f"{args.width}x{args.blocks}_{args.optimizer}" + (
        f"_pca-{args.pca_mode}" if args.pca_mode != "none" else "")
    out_dir = Path(args.out) if args.out else Path("runs") / f"{ds_name}_{args.net}_{tag}"
    start_epoch, global_step, best_val, bad_epochs = 0, 0, float("inf"), 0
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        prep = Path(args.prep_dir) if args.prep_dir else Path(args.data_dir) / "prep"
        for f in (f"norm_{args.net}.json", f"norm_{args.net}.targets.json",
                  f"norm_{args.net}.npz", f"pca_{args.net}.npz"):
            if (prep / f).exists():
                shutil.copy2(prep / f, out_dir / f)
        (out_dir / "config.json").write_text(json.dumps(
            {**vars(args), "model": model_cfg, "d_in": ispec.d_in,
             "d_out": ospec.d_out, "world": world,
             "fingerprint": man.fingerprint()},
            indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        csv_path = out_dir / "metrics.csv"
        csv_new = not csv_path.exists() or not args.resume
        csv_f = open(csv_path, "w" if csv_new else "a", newline="")
        csv_w = csv.writer(csv_f)
        if csv_new:
            csv_w.writerow(["epoch", "step", "lr", "train_loss", "val_loss",
                            "best_val", "seconds"])

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=True)
        raw_model.load_state_dict(ck["model_raw"])
        opt.load_state_dict(ck["opt"])
        ema.shadow = {k: v.to(device) for k, v in ck["model_ema"].items()}
        start_epoch = ck["epoch"] + 1
        global_step = ck["global_step"]
        best_val = ck.get("best_val", float("inf"))
        if is_main:
            print(f"[resume] {args.resume} -> epoch {start_epoch}")

    def save(path: Path, epoch: int, val_loss: float):
        torch.save({"model_ema": ema.state_dict_for(raw_model),
                    "model_raw": raw_model.state_dict(),
                    "opt": opt.state_dict(),
                    "config": model_cfg, "net": args.net,
                    "args": {k: str(v) for k, v in vars(args).items()},
                    "fingerprint": man.fingerprint(),
                    "epoch": epoch, "global_step": global_step,
                    "best_val": best_val, "val_loss": val_loss}, path)

    # ------------------------------------------------------------ loop --
    stop = torch.zeros(1, device=device)
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        loss_sum, nb = 0.0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.set_lr_scale(lr_scale(global_step, total_steps,
                                      args.warmup_frac, args.min_lr_frac))
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_bf16):
                pred = model(x)
            pred = pred.float()
            target = y if v_t is None else (y - vm_t) @ v_t
            if args.huber_delta > 0:
                loss = F.huber_loss(pred, target, delta=args.huber_delta)
            else:
                loss = F.mse_loss(pred, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(),
                                               args.grad_clip)
            opt.step()
            ema.update(raw_model)
            global_step += 1
            loss_sum += loss.item()
            nb += 1

        val_loss = float("nan")
        if is_main:
            val_loss = validate(model_cfg, ema, val_X, val_Y, device, pca)
            improved = val_loss < best_val - 1e-8
            if improved:
                best_val = val_loss
                bad_epochs = 0
                save(out_dir / "best.pt", epoch, val_loss)
            else:
                bad_epochs += 1
            save(out_dir / "last.pt", epoch, val_loss)
            dt = time.time() - t0
            print(f"epoch {epoch:4d}  lr {opt.lr:.2e}  "
                  f"train {loss_sum / max(nb, 1):.5e}  val {val_loss:.5e}"
                  f"{'  *' if improved else ''}  ({dt:.1f}s)")
            csv_w.writerow([epoch, global_step, f"{opt.lr:.3e}",
                            f"{loss_sum / max(nb, 1):.6e}", f"{val_loss:.6e}",
                            f"{best_val:.6e}", f"{dt:.1f}"])
            csv_f.flush()
            stop[0] = float(bad_epochs >= args.patience)
        if world > 1:
            dist.broadcast(stop, src=0)
        if stop.item():
            if is_main:
                print(f"[early-stop] {args.patience} epochs 无改善，"
                      f"best val {best_val:.5e}")
            break

    if is_main:
        csv_f.close()
        print(f"[done] best val {best_val:.5e} -> {out_dir / 'best.pt'}")
        print(f"       评估: python scripts/evaluate.py --run-dir {out_dir} "
              f"--data-dir <独立 random 测试集>")
        if args.export_safetensors:
            from scripts.export_safetensors import export_run, self_check
            out = export_run(out_dir, Path(args.export_safetensors), "best.pt")
            print(f"[export] -> {out}")
            self_check(out_dir, out, "best.pt")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
