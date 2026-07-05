"""优化器构建：AdamW（默认）或 Muon 混合方案。

Muon 只作用于隐藏层 2D 权重矩阵（残差块内的 Linear.weight）；
输入层、输出头、LayerNorm 增益、所有 bias 仍用 AdamW——这是 Muon 的
标准用法（首末层与 1D 参数不做 Newton–Schulz 正交化）。

优先用 torch.optim.Muon（PyTorch ≥ 2.9 原生），带 match_rms_adamw 学习率
匹配时可直接复用 AdamW 的 lr/wd；老版本 torch 回退到内置 SimpleMuon
（Newton–Schulz 5 步 + Moonlight 的 0.2·√max(m,n) RMS 匹配缩放，
同样直接复用 AdamW 超参）。
"""
from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn


def _newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """迭代正交化 momentum 矩阵（Keller Jordan 的五阶 Newton–Schulz 系数）。"""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class SimpleMuon(torch.optim.Optimizer):
    """单机版 Muon 回退实现（torch < 2.9 时使用）。

    更新量做 0.2·√max(m,n) 缩放以匹配 AdamW 的典型 update RMS（Moonlight
    'Muon is Scalable for LLM Training' 的做法），因此 lr / weight_decay
    直接沿用 AdamW 的取值。
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.0,
                 momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            mom, nesterov = group["momentum"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("SimpleMuon 只接受 2D 权重矩阵")
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                u = g.add(buf, alpha=mom) if nesterov else buf
                o = _newton_schulz5(u, group["ns_steps"]).to(p.dtype)
                scale = 0.2 * math.sqrt(max(p.shape[0], p.shape[1]))
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(o, alpha=-lr * scale)
        return loss


class OptimizerGroup:
    """把 1–2 个优化器打包成单一接口，并支持按比例调 lr（cosine 调度用）。"""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers
        for opt in self.optimizers:
            for g in opt.param_groups:
                g.setdefault("initial_lr", g["lr"])

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def set_lr_scale(self, scale: float):
        for opt in self.optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale

    @property
    def lr(self) -> float:
        return self.optimizers[0].param_groups[0]["lr"]

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, states):
        for opt, s in zip(self.optimizers, states):
            opt.load_state_dict(s)


def split_params(model: nn.Module):
    """隐藏层 2D 权重（残差块 Linear.weight）→ Muon；其余 → AdamW。"""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and name.startswith("blocks."):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


def build_optimizer(model: nn.Module, name: str = "adamw",
                    lr: float = 1e-3, weight_decay: float = 1e-4) -> OptimizerGroup:
    name = name.lower()
    if name == "adamw":
        return OptimizerGroup([torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)])
    if name != "muon":
        raise ValueError(f"未知优化器: {name}（可选 adamw / muon）")

    muon_params, adamw_params = split_params(model)
    if not muon_params:
        print("[optim] 模型无隐藏层 2D 权重，muon 退化为纯 AdamW")
        return OptimizerGroup([torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)])

    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is not None:
        kwargs = dict(lr=lr, weight_decay=weight_decay)
        sig = inspect.signature(muon_cls.__init__).parameters
        if "adjust_lr_fn" in sig:
            kwargs["adjust_lr_fn"] = "match_rms_adamw"  # 直接复用 AdamW 的 lr
        if "momentum" in sig:
            kwargs["momentum"] = 0.95
        if "nesterov" in sig:
            kwargs["nesterov"] = True
        muon = muon_cls(muon_params, **kwargs)
        print(f"[optim] torch.optim.Muon x{len(muon_params)} 张量 "
              f"+ AdamW x{len(adamw_params)} 张量（match_rms_adamw="
              f"{'on' if 'adjust_lr_fn' in kwargs else 'n/a'}）")
    else:
        muon = SimpleMuon(muon_params, lr=lr, weight_decay=weight_decay)
        print(f"[optim] torch {torch.__version__} 无原生 Muon，"
              f"使用内置 SimpleMuon x{len(muon_params)} 张量 "
              f"+ AdamW x{len(adamw_params)} 张量")
    adamw = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay)
    return OptimizerGroup([muon, adamw])
