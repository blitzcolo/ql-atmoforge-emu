"""输入/输出变换（ModModel.md §5、§6）。

输入金规则：用 manifest 采样范围做解析归一化，不用数据统计量。
    uniform      x̂ = 2(x−lo)/(hi−lo) − 1
    log_uniform  同上作用于 ln x
    天顶角       先 cos 再 min-max
    相对方位角   取 cos(Δφ)（±Δφ 镜像对称）再 min-max
    离散维       one-hot（按 values 声明顺序）
    常量维       剔除

输出：逐通道 μ/σ 标准化 + 方差下限 max(σ_ν, floor_frac·median(σ))；
δ 目标先钳 [0, clamp_max]；辐亮度目标对动态范围 >10³ 的通道先 log(L+ε)。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .manifest import Manifest, dist_bounds, slant_range_km

# ------------------------------------------------------------------ inputs --

# 三网的物理不变性裁剪（ModModel.md §5.3）
def excluded_input_names(man: Manifest, net: str) -> set[str]:
    if net == "tau":
        # τ 与太阳无关（IEMSCT=0 本不看太阳）
        return {"sun_zenith_deg", "sun_rel_azimuth_deg"}
    if net == "ldown":
        if man.is_slant:
            # 斜程 ldown 在地面定点仰视，与观测几何无关
            return {"h1_km", "view_zenith_deg"}
        # 横程 ldown 在 h1 高度仰视：保留 h1，剔除视线距离
        return {"range_km"}
    return set()


@dataclass
class InputSpec:
    """可 JSON 序列化的输入变换序列。entry.kind:
    linear / log — min-max 到 [-1,1]；cos_deg — 先 cos(deg) 再 min-max；
    onehot — 按 values 展开。col 是 params.npy 的列号。"""
    entries: list[dict] = field(default_factory=list)

    @property
    def d_in(self) -> int:
        return sum(len(e["values"]) if e["kind"] == "onehot" else 1
                   for e in self.entries)

    @property
    def feature_names_out(self) -> list[str]:
        names = []
        for e in self.entries:
            if e["kind"] == "onehot":
                names += [f"{e['name']}={int(v)}" for v in e["values"]]
            else:
                names.append(e["name"])
        return names

    def apply(self, P: np.ndarray) -> np.ndarray:
        """P: params 数组 [N, n_cols] → X float32 [N, d_in]。"""
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        cols: list[np.ndarray] = []
        for e in self.entries:
            x = P[:, e["col"]]
            kind = e["kind"]
            if kind == "onehot":
                for v in e["values"]:
                    cols.append((np.abs(x - v) < 1e-9).astype(np.float64))
                continue
            if kind == "log":
                x = np.log(np.maximum(x, 1e-300))
            elif kind == "cos_deg":
                x = np.cos(np.radians(x))
            lo, hi = e["lo"], e["hi"]
            cols.append(2.0 * (x - lo) / (hi - lo) - 1.0)
        return np.stack(cols, axis=1).astype(np.float32)

    def to_json(self) -> dict:
        return {"entries": self.entries, "d_in": self.d_in,
                "feature_names_out": self.feature_names_out}

    @classmethod
    def from_json(cls, d: dict) -> "InputSpec":
        return cls(entries=d["entries"])


def build_input_spec(man: Manifest, net: str) -> InputSpec:
    """从 manifest 解析构建输入变换（确定性，与数据内容无关）。"""
    excl = excluded_input_names(man, net)
    entries: list[dict] = []
    sampled = man.sampled
    P = len(man.feature_names)
    gcol = {name: P - 4 + j for j, name in
            enumerate(["h1_km", "h2_km", "cos_view_zenith", "range_km"])}

    for i, (name, dist) in enumerate(sampled.items()):
        if name in excl:
            continue
        if name == "view_zenith_deg":
            continue  # 用几何块的 cos_view_zenith 表达（有界、光滑、正比气团）
        if "values" in dist:
            vals = [float(v) for v in dist["values"]]
            if len(vals) < 2:
                continue  # 单值离散维是常量
            entries.append({"kind": "onehot", "name": name, "col": i,
                            "values": vals})
        elif name == "sun_zenith_deg":
            lo, hi = dist_bounds(dist)
            clo, chi = math.cos(math.radians(hi)), math.cos(math.radians(lo))
            entries.append({"kind": "cos_deg", "name": "cos_sun_zenith",
                            "col": i, "lo": clo, "hi": chi})
        elif name == "sun_rel_azimuth_deg":
            lo, hi = dist_bounds(dist)
            # cos 在区间上的界：端点 + 区间内的 0°/±180° 极值点
            cands = [lo, hi] + [t for t in (-180.0, 0.0, 180.0) if lo < t < hi]
            cvals = [math.cos(math.radians(t)) for t in cands]
            clo, chi = min(cvals), max(cvals)
            if chi - clo < 1e-9:
                continue  # 方位角范围退化为常量
            entries.append({"kind": "cos_deg", "name": "cos_sun_rel_azimuth",
                            "col": i, "lo": clo, "hi": chi})
        elif "log_uniform" in dist:
            lo, hi = dist_bounds(dist)
            entries.append({"kind": "log", "name": name, "col": i,
                            "lo": math.log(lo), "hi": math.log(hi)})
        else:  # uniform
            lo, hi = dist_bounds(dist)
            if hi - lo < 1e-12:
                continue
            entries.append({"kind": "linear", "name": name, "col": i,
                            "lo": lo, "hi": hi})

    # 几何块：h2 恒冗余（斜程≡0 / 横程≡h1）；h1、横程 range 与采样块重复。
    if man.is_slant and net != "ldown":
        th_lo, th_hi = man.param_range("view_zenith_deg")
        if th_hi - th_lo > 1e-9:
            clo = math.cos(math.radians(th_hi))
            chi = math.cos(math.radians(th_lo))
            entries.append({"kind": "linear", "name": "cos_view_zenith",
                            "col": gcol["cos_view_zenith"], "lo": clo, "hi": chi})
        h1_lo, h1_hi = man.param_range("h1_km")
        # 派生 range 的解析界：range 随 h1 单调增、随 θ→180° 单调减
        r_lo = slant_range_km(h1_lo, th_hi)
        r_hi = slant_range_km(h1_hi, th_lo)
        if r_hi / max(r_lo, 1e-12) > 1.0 + 1e-9:
            entries.append({"kind": "log", "name": "range_km",
                            "col": gcol["range_km"],
                            "lo": math.log(r_lo), "hi": math.log(r_hi)})

    if not entries:
        raise ValueError(f"net={net}: 无可用输入维（全为常量？）")
    return InputSpec(entries=entries)


# ----------------------------------------------------------------- outputs --

def default_targets(man: Manifest, net: str) -> list[tuple[str, str, str]]:
    """(block, column, kind) 列表。kind: delta / radiance。ModModel.md §6.1。"""
    band = man.band["name"]
    if net == "tau":
        return [("tau", "LOG_TOTAL", "delta")]
    if net == "lpath":
        if band == "mwir":
            # 日光/热交叉区拆分量头：夜间下游置零 SOL_SCAT 即可
            return [("lpath", "PTH_THRML", "radiance"),
                    ("lpath", "SOL_SCAT", "radiance")]
        return [("lpath", "TOTAL_RAD", "radiance")]
    if net == "ldown":
        if not man.band["thermal"]:
            raise ValueError(f"波段 {band} 非 thermal，数据集无 ldown 块")
        return [("ldown", "TOTAL_RAD", "radiance")]
    raise ValueError(f"未知 net: {net}")


@dataclass
class TargetRow:
    block: str
    column: str
    kind: str                     # delta | radiance
    col_index: int                # 在 merge 后 npy 的列号
    clamp_max: float | None = None
    # fit 后填充，均为 [K]
    log_mask: np.ndarray | None = None
    log_eps: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def transform(self, y: np.ndarray) -> np.ndarray:
        """y [..., K] 物理值 → 标准化值。"""
        y = np.asarray(y, dtype=np.float64)
        if self.kind == "delta":
            y = np.clip(y, 0.0, self.clamp_max)
        if self.log_mask is not None and self.log_mask.any():
            # np.where 对未选中的线性分支也求值，log(0+0) 的 -inf 会被丢弃
            # 但仍触发 RuntimeWarning——按掩码静音
            with np.errstate(divide="ignore"):
                y = np.where(self.log_mask,
                             np.log(np.maximum(y, 0.0) + self.log_eps), y)
        return (y - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        y = np.asarray(z, dtype=np.float64) * self.std + self.mean
        if self.log_mask is not None and self.log_mask.any():
            y = np.where(self.log_mask, np.exp(y) - self.log_eps, y)
        return y


@dataclass
class OutputSpec:
    rows: list[TargetRow]
    K: int

    @property
    def d_out(self) -> int:
        return len(self.rows) * self.K

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Y [T, K] 或 [N, T, K] → 标准化并展平末两维，float32。"""
        Y = np.asarray(Y, dtype=np.float64)
        single = Y.ndim == 2
        if single:
            Y = Y[None]
        out = np.empty(Y.shape, dtype=np.float64)
        for t, r in enumerate(self.rows):
            out[:, t] = r.transform(Y[:, t])
        out = out.reshape(Y.shape[0], -1).astype(np.float32)
        return out[0] if single else out

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        """[N, T*K] 标准化值 → [N, T, K] 物理值 float64。"""
        Z = np.asarray(Z, dtype=np.float64).reshape(-1, len(self.rows), self.K)
        out = np.empty(Z.shape, dtype=np.float64)
        for t, r in enumerate(self.rows):
            out[:, t] = r.inverse(Z[:, t])
        return out

    # -------------------------------------------------------- persistence --
    def save(self, json_path: Path, npz_path: Path) -> None:
        meta = {"K": self.K, "rows": [
            {"block": r.block, "column": r.column, "kind": r.kind,
             "col_index": r.col_index, "clamp_max": r.clamp_max}
            for r in self.rows]}
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        arrays = {}
        for t, r in enumerate(self.rows):
            arrays[f"{t}.mean"] = r.mean
            arrays[f"{t}.std"] = r.std
            arrays[f"{t}.log_mask"] = r.log_mask
            arrays[f"{t}.log_eps"] = r.log_eps
        np.savez_compressed(npz_path, **arrays)

    @classmethod
    def load(cls, json_path: Path, npz_path: Path) -> "OutputSpec":
        meta = json.loads(Path(json_path).read_text(encoding="utf-8"))
        z = np.load(npz_path)
        rows = []
        for t, rm in enumerate(meta["rows"]):
            rows.append(TargetRow(
                block=rm["block"], column=rm["column"], kind=rm["kind"],
                col_index=rm["col_index"], clamp_max=rm["clamp_max"],
                log_mask=z[f"{t}.log_mask"], log_eps=z[f"{t}.log_eps"],
                mean=z[f"{t}.mean"], std=z[f"{t}.std"]))
        return cls(rows=rows, K=meta["K"])


def fit_output_spec(man: Manifest, targets: list[tuple[str, str, str]],
                    train_idx: np.ndarray, *, delta_clamp: float = 20.0,
                    log_dynrange: float = 1e3, floor_frac: float = 0.01,
                    chunk: int = 2048, verbose: bool = True) -> OutputSpec:
    """流式（mmap 分块）在训练子集上拟合输出统计量。"""
    K = man.K
    rows: list[TargetRow] = []
    train_idx = np.sort(np.asarray(train_idx))
    for block, column, kind in targets:
        r = TargetRow(block=block, column=column, kind=kind,
                      col_index=man.col_index(block, column),
                      clamp_max=delta_clamp if kind == "delta" else None)
        arr = man.npy(block)  # mmap [N, C, K]

        def chunks():
            for s in range(0, len(train_idx), chunk):
                idx = train_idx[s:s + chunk]
                y = np.asarray(arr[idx, r.col_index, :], dtype=np.float64)
                if kind == "delta":
                    y = np.clip(y, 0.0, delta_clamp)
                yield y

        # pass 1: 每通道 max 与最小正值 → log 决策（仅辐亮度）
        if kind == "radiance":
            vmax = np.full(K, -np.inf)
            vminpos = np.full(K, np.inf)
            for y in chunks():
                vmax = np.maximum(vmax, y.max(axis=0))
                ypos = np.where(y > 0, y, np.inf)
                vminpos = np.minimum(vminpos, ypos.min(axis=0))
            alive = vmax > 0
            dyn = np.where(alive, vmax / np.maximum(vminpos, 1e-300), 0.0)
            r.log_mask = alive & (dyn > log_dynrange)
            r.log_eps = np.where(r.log_mask, 1e-4 * vmax, 0.0)
        else:
            r.log_mask = np.zeros(K, dtype=bool)
            r.log_eps = np.zeros(K)

        # pass 2: 变换后逐通道 μ/σ
        n = 0
        s1 = np.zeros(K)
        s2 = np.zeros(K)
        for y in chunks():
            if r.log_mask.any():
                with np.errstate(divide="ignore"):
                    y = np.where(r.log_mask,
                                 np.log(np.maximum(y, 0.0) + r.log_eps), y)
            n += y.shape[0]
            s1 += y.sum(axis=0)
            s2 += (y * y).sum(axis=0)
        mean = s1 / n
        var = np.maximum(s2 / n - mean * mean, 0.0)
        std = np.sqrt(var)
        # 方差下限（§6.2）按变换分组各算各的：log 通道的 σ 是 ln 空间的
        # O(1)，线性通道的 σ 是物理单位（辐亮度低到 1e-8）。混成一个中位数
        # 时，log 多数会把下限抬到 0.01×O(1)，线性通道整组触底——mwir ldown
        # 实测 448 个线性通道全部被压成 σ=9e-3（真实 σ 的 ~1e6 倍），标签
        # 标准化后塌缩成常数，网络在这些通道上学不到任何东西。
        r.mean = mean
        r.std = std.copy()
        stats = []
        for name_g, grp in (("log", r.log_mask), ("lin", ~r.log_mask)):
            if not grp.any():
                continue
            med = np.median(std[grp])
            floor = max(floor_frac * med, 1e-30)
            r.std[grp] = np.maximum(std[grp], floor)
            stats.append(f"{name_g} {int(grp.sum())}ch σ中位 {med:.3e} "
                         f"触底 {int((std[grp] < floor).sum())}")
        if verbose:
            print(f"  [{block}.{column}] n={n}  " + "; ".join(stats))
        rows.append(r)
    return OutputSpec(rows=rows, K=K)
