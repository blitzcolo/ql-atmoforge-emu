"""读取 ql-atmoforge merge 产出的数据集目录（manifest.json + *.npy）。

特征向量布局（与生成端 src/params.cpp 对齐）：
    [采样维按 manifest["sampled"] 声明顺序] + [h1_km, h2_km, cos_view_zenith, range_km]
采样块与几何块可能重名（如 h1_km 既被采样又在几何块），按位置区分：
最后 4 列恒为几何块。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GEOMETRY_FEATURES = ["h1_km", "h2_km", "cos_view_zenith", "range_km"]
EARTH_RADIUS_KM = 6371.0  # 必须与 ql-atmoforge src/params.cpp kEarthRadiusKm 一致

STATUS_OK = 0
STATUS_PARTIAL = 1
STATUS_FAILED = 2


def slant_range_km(h1_km: float, view_zenith_deg: float) -> float:
    """球面无折射斜程距离，逐行复刻生成端 ParamSpace::resolve()（含擦地钳制）。"""
    r1 = EARTH_RADIUS_KM + h1_km
    theta_min = 180.0 - math.degrees(math.asin(EARTH_RADIUS_KM / r1))
    theta = max(view_zenith_deg, theta_min + 0.05)
    mu = math.cos(math.radians(theta))
    disc = r1 * r1 * mu * mu - (r1 * r1 - EARTH_RADIUS_KM * EARTH_RADIUS_KM)
    return -r1 * mu - math.sqrt(max(disc, 0.0))


def dist_bounds(dist: dict) -> tuple[float, float]:
    """任意分布字典的 (min, max)。"""
    if "values" in dist:
        return float(min(dist["values"])), float(max(dist["values"]))
    if "uniform" in dist:
        lo, hi = dist["uniform"]
        return float(lo), float(hi)
    if "log_uniform" in dist:
        lo, hi = dist["log_uniform"]
        return float(lo), float(hi)
    raise ValueError(f"unknown dist: {dist}")


@dataclass
class Manifest:
    data_dir: Path
    raw: dict

    @classmethod
    def load(cls, data_dir: str | Path) -> "Manifest":
        data_dir = Path(data_dir)
        mf = data_dir / "manifest.json"
        if not mf.exists():
            raise FileNotFoundError(
                f"{mf} 不存在——data_dir 应指向 `ql-atmoforge merge` 的 out_dir")
        raw = json.loads(mf.read_text(encoding="utf-8"))
        fmt = raw.get("format", "")
        if not str(fmt).startswith("ql-atmoforge-dataset"):
            raise ValueError(f"未知数据集格式: {fmt!r}")
        m = cls(data_dir=data_dir, raw=raw)
        m._validate()
        return m

    def _validate(self) -> None:
        names = self.feature_names
        if names[-4:] != GEOMETRY_FEATURES:
            raise ValueError(f"特征布局异常，末 4 列应为几何块，实际: {names[-4:]}")
        if names[: len(names) - 4] != list(self.sampled.keys()):
            raise ValueError("feature_names 采样块与 manifest['sampled'] 顺序不一致")

    # ---------------------------------------------------------- properties --
    @property
    def feature_names(self) -> list[str]:
        return list(self.raw["feature_names"])

    @property
    def sampled(self) -> dict[str, dict]:
        return self.raw["sampled"]

    @property
    def fixed_effective(self) -> dict[str, float]:
        return self.raw.get("fixed_effective", {})

    @property
    def band(self) -> dict:
        return self.raw["band"]

    @property
    def K(self) -> int:
        return int(self.band["K"])

    @property
    def path_type(self) -> str:
        return self.raw["path_type"]

    @property
    def is_slant(self) -> bool:
        return self.path_type != "horizontal"

    @property
    def n_present(self) -> int:
        return int(self.raw["n_present"])

    @property
    def sampler(self) -> str:
        return self.raw.get("sampler", "?")

    @property
    def blocks(self) -> list[str]:
        return [b for b in ("tau", "lpath", "ldown") if b in self.raw["arrays"]]

    # ------------------------------------------------------------- helpers --
    def param_range(self, name: str) -> tuple[float, float]:
        """参数的取值范围；固定参数返回 (v, v)。"""
        if name in self.sampled:
            return dist_bounds(self.sampled[name])
        if name in self.fixed_effective:
            v = float(self.fixed_effective[name])
            return v, v
        raise KeyError(f"参数 {name} 既不在 sampled 也不在 fixed_effective")

    def block_columns(self, block: str) -> list[str]:
        return list(self.raw["arrays"][block]["columns"])

    def col_index(self, block: str, column: str) -> int:
        cols = self.block_columns(block)
        if column not in cols:
            raise KeyError(
                f"{block}.npy 不含列 {column}（merge 白名单为 {cols}；"
                f"shard 存全列，改 columns 重新 merge 即可）")
        return cols.index(column)

    def npy(self, name: str, mmap: bool = True) -> np.ndarray:
        return np.load(self.data_dir / f"{name}.npy",
                       mmap_mode="r" if mmap else None)

    def wavenumber(self) -> np.ndarray:
        return np.asarray(self.npy("wavenumber", mmap=False), dtype=np.float64)

    def fingerprint(self) -> dict:
        """训练/评估数据集兼容性检查用的最小指纹。"""
        return {
            "band": {k: self.band[k] for k in
                     ("name", "v1_cm", "v2_cm", "dv_cm", "fwhm_cm", "K", "thermal")},
            "path_type": self.path_type,
            "feature_names": self.feature_names,
            "sampled": self.sampled,
        }

    def check_compatible(self, other_fp: dict) -> None:
        fp = self.fingerprint()
        for key in ("band", "path_type", "feature_names"):
            if fp[key] != other_fp[key]:
                raise ValueError(
                    f"数据集不兼容：{key} 不一致\n  训练: {other_fp[key]}\n  当前: {fp[key]}")
