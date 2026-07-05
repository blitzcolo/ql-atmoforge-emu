"""torch Dataset：params 常驻内存（很小），光谱块 mmap 或 --preload 进 RAM。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import Manifest
from .transforms import InputSpec, OutputSpec


class SpectraDataset(Dataset):
    """返回 (X float32 [d_in], y float32 [T*K])，y 已标准化。

    mmap 模式下 npy 句柄在每个 DataLoader worker 内惰性打开（fork 安全）。
    """

    def __init__(self, man: Manifest, indices: np.ndarray,
                 input_spec: InputSpec, output_spec: OutputSpec,
                 preload: bool = False):
        self.man = man
        self.indices = np.asarray(indices, dtype=np.int64)
        self.output_spec = output_spec
        params = np.asarray(man.npy("params", mmap=False))
        self.X = torch.from_numpy(input_spec.apply(params[self.indices]))
        self._blocks: dict[str, np.ndarray] | None = None
        self._preloaded: torch.Tensor | None = None
        if preload:
            Y = np.empty((len(self.indices), len(output_spec.rows), man.K),
                         dtype=np.float32)
            order = np.argsort(self.indices)  # mmap 顺序读
            for t, r in enumerate(output_spec.rows):
                arr = man.npy(r.block)
                for s in range(0, len(order), 4096):
                    o = order[s:s + 4096]
                    Y[o, t] = arr[self.indices[o], r.col_index, :]
            self._preloaded = torch.from_numpy(
                output_spec.transform(Y.astype(np.float64)))

    def _open(self) -> dict[str, np.ndarray]:
        if self._blocks is None:
            self._blocks = {r.block: self.man.npy(r.block)
                            for r in self.output_spec.rows}
        return self._blocks

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        if self._preloaded is not None:
            return self.X[i], self._preloaded[i]
        blocks = self._open()
        rows = self.output_spec.rows
        y = np.empty((len(rows), self.man.K), dtype=np.float64)
        for t, r in enumerate(rows):
            y[t] = blocks[r.block][self.indices[i], r.col_index, :]
        return self.X[i], torch.from_numpy(self.output_spec.transform(y))


# ------------------------------------------------------------- prep loader --

def load_prep(data_dir: str | Path, net: str, prep_dir: str | Path | None = None):
    """读取 prepare_data.py 的产物。返回 (man, input_spec, output_spec, splits, pca)。"""
    data_dir = Path(data_dir)
    prep = Path(prep_dir) if prep_dir else data_dir / "prep"
    man = Manifest.load(data_dir)
    norm = json.loads((prep / f"norm_{net}.json").read_text(encoding="utf-8"))
    input_spec = InputSpec.from_json(norm["input"])
    output_spec = OutputSpec.load(prep / f"norm_{net}.targets.json",
                                  prep / f"norm_{net}.npz")
    splits = json.loads((prep / "splits.json").read_text(encoding="utf-8"))
    idx = {k: np.load(prep / f"{k}_idx.npy")
           for k in ("train", "val", "test") if (prep / f"{k}_idx.npy").exists()}
    pca_path = prep / f"pca_{net}.npz"
    pca = np.load(pca_path) if pca_path.exists() else None
    return man, input_spec, output_spec, {**splits, "idx": idx}, pca
