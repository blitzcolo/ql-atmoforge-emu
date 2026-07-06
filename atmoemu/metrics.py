"""评估指标（ModModel.md §7.3）：逐通道 NRMSE、τ/δ 绝对误差、亮温误差、P99。"""
from __future__ import annotations

import numpy as np

from .transforms import OutputSpec

# Planck 常数，单位制与 tape7 辐亮度一致：L [W cm-2 sr-1 / cm-1]，ν [cm-1]
_C1 = 1.191042972e-12   # 2 h c^2  [W cm-2 sr-1 (cm-1)^-4]
_C2 = 1.4387769         # h c / k  [cm K]


def brightness_temperature(L: np.ndarray, nu_cm: np.ndarray) -> np.ndarray:
    """逆 Planck：T_b = c2·ν / ln(1 + c1·ν³/L)。L<=0 处返回 NaN。"""
    L = np.asarray(L, dtype=np.float64)
    nu = np.asarray(nu_cm, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _C2 * nu / np.log1p(_C1 * nu ** 3 / np.where(L > 0, L, np.nan))
    return t


def _p99(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 99)) if a.size else float("nan")


def compute_metrics(z_true: np.ndarray, z_pred: np.ndarray,
                    output_spec: OutputSpec, nu_cm: np.ndarray,
                    thermal: bool) -> dict:
    """z_*: 标准化空间 [N, T*K]。返回标量指标 dict + 每行物理量误差。"""
    err_std = z_pred - z_true
    K = output_spec.K
    T = len(output_spec.rows)
    per_ch_rmse = np.sqrt((err_std ** 2).mean(axis=0))          # ≡ NRMSE(σ_train)
    out = {
        "mse_std": float((err_std ** 2).mean()),
        "nrmse_mean": float(per_ch_rmse.mean()),
        "nrmse_p99": _p99(per_ch_rmse),
        "nrmse_max": float(per_ch_rmse.max()),
    }

    y_true = output_spec.inverse(z_true)    # [N, T, K] 物理值
    y_pred = output_spec.inverse(z_pred)
    for t, r in enumerate(output_spec.rows):
        tag = f"{r.block}.{r.column}"
        a, b = y_true[:, t], y_pred[:, t]
        if r.kind == "delta":
            d_err = np.abs(b - a)
            tau_err = np.abs(np.exp(-b) - np.exp(-a))
            out[f"{tag}/delta_mae"] = float(d_err.mean())
            out[f"{tag}/delta_p99"] = _p99(d_err)
            out[f"{tag}/tau_mae"] = float(tau_err.mean())        # δ 绝对误差 ≈ τ 相对误差
            out[f"{tag}/tau_p99"] = _p99(tau_err)
        else:
            # 近零保护：辐亮度真值在吸收带芯/冷背景处 ≈0，相对误差与逆
            # Planck 亮温在零附近发散，会把无信号元素的数值噪声放大成
            # 天文数字（MWIR 实测 rel_mae 上万、bt_mae 数百 K，而标准化
            # NRMSE 同时只有百分之几）。口径改为"有信号处的精度"：
            #  - 预测钳非负（部署契约相同，infer.py 亦然）；
            #  - 通道尺度取该通道 |L| 的 99 分位（对圆日等离群样本稳健），
            #    只统计 L_true 超过通道尺度 1% 且通道尺度不低于整带尺度
            #    1e-3 的元素（后者剔除全零死通道），覆盖率记
            #    rad_valid_frac / bt_valid_frac。
            ref = float(np.percentile(np.abs(a), 99.9)) + 1e-300
            cs = np.percentile(np.abs(a), 99, axis=0)          # [K] 通道尺度
            valid = (np.abs(a) > 1e-2 * cs[None, :]) & (cs > 1e-3 * ref)[None, :]
            bp = np.maximum(b, 0.0)
            rel = np.where(valid,
                           np.abs(bp - a) / np.maximum(np.abs(a), 1e-300),
                           np.nan)
            out[f"{tag}/rad_rel_mae"] = float(np.nanmean(rel))
            out[f"{tag}/rad_rel_p99"] = _p99(rel)
            out[f"{tag}/rad_valid_frac"] = float(valid.mean())
            if thermal and r.column in ("TOTAL_RAD", "PTH_THRML"):
                bt_t = brightness_temperature(np.where(valid, a, np.nan), nu_cm)
                bt_p = brightness_temperature(np.where(valid, bp, np.nan), nu_cm)
                dt = np.abs(bt_p - bt_t)
                out[f"{tag}/bt_mae_K"] = float(np.nanmean(dt))
                out[f"{tag}/bt_p99_K"] = _p99(dt)
                out[f"{tag}/bt_valid_frac"] = float(valid.mean())
    out["per_channel_nrmse"] = per_ch_rmse.reshape(T, K)
    return out


def format_metrics(m: dict) -> str:
    lines = []
    for k, v in m.items():
        if k == "per_channel_nrmse":
            continue
        lines.append(f"  {k:36s} {v:.5g}")
    return "\n".join(lines)
