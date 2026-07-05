"""残差 MLP（ModModel.md §7.1）：

    输入 → Linear(width) → N × [Linear → LayerNorm → SiLU → Linear, skip]
        → 输出头（dense / PCA 初始化线性解码头 / PCA 系数头）
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.norm(self.fc1(x))))


class ResMLP(nn.Module):
    """pca_mode:
      "none"  — dense 头 Linear(width, d_out)
      "head"  — Linear(width, n_pc) → Linear(n_pc, d_out)，解码层用 PCA 基初始化，
                随整网端到端微调（推荐，ModModel.md §6.3）
      "coeff" — 输出 n_pc 维系数，损失在系数空间（Parseval ⇒ ≡ 谱空间 MSE），
                评估时用固定 PCA 基解码
    """

    def __init__(self, d_in: int, d_out: int, width: int = 256, blocks: int = 4,
                 pca_mode: str = "none", pca_basis: np.ndarray | None = None,
                 pca_mean: np.ndarray | None = None):
        super().__init__()
        self.config = {"d_in": d_in, "d_out": d_out, "width": width,
                       "blocks": blocks, "pca_mode": pca_mode,
                       "n_pc": 0 if pca_basis is None else int(pca_basis.shape[1])}
        self.stem = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(blocks))
        if pca_mode == "none":
            self.head = nn.Linear(width, d_out)
        else:
            if pca_basis is None:
                raise ValueError(f"pca_mode={pca_mode} 需要 PCA 基")
            V = torch.as_tensor(np.asarray(pca_basis), dtype=torch.float32)
            if V.shape[0] != d_out:
                raise ValueError(f"PCA 基形状 {tuple(V.shape)} 与 d_out={d_out} 不符")
            n_pc = V.shape[1]
            self.proj = nn.Linear(width, n_pc)
            if pca_mode == "head":
                dec = nn.Linear(n_pc, d_out)
                with torch.no_grad():
                    dec.weight.copy_(V)      # [d_out, n_pc]
                    if pca_mean is not None:
                        dec.bias.copy_(torch.as_tensor(
                            np.asarray(pca_mean), dtype=torch.float32))
                    else:
                        dec.bias.zero_()
                self.head = dec
            elif pca_mode == "coeff":
                self.head = None             # 输出即系数
            else:
                raise ValueError(f"未知 pca_mode: {pca_mode}")

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        if self.config["pca_mode"] == "none":
            return self.head(h)
        c = self.proj(h)
        return c if self.head is None else self.head(c)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: dict, pca_basis: np.ndarray | None = None,
                pca_mean: np.ndarray | None = None) -> ResMLP:
    return ResMLP(d_in=cfg["d_in"], d_out=cfg["d_out"], width=cfg["width"],
                  blocks=cfg["blocks"], pca_mode=cfg["pca_mode"],
                  pca_basis=pca_basis, pca_mean=pca_mean)


class EMA:
    """权重指数滑动平均；评估/导出用 EMA 权重（ModModel.md §7.2）。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        # 预热斜坡：训练早期有效 decay 从小值渐升到目标值，
        # 否则前几千步 EMA 仍近似初始权重，val 指标失真
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def state_dict_for(self, model: nn.Module) -> dict:
        out = {}
        for k, v in model.state_dict().items():
            out[k] = self.shadow[k].to(v.dtype)
        return out
