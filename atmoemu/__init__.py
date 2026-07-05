"""atmoemu — ql-atmoforge 数据集的 MODTRAN 代理网络训练包。

数据格式对齐 ql-atmoforge merge 产出（format = ql-atmoforge-dataset-v1）：
manifest.json + params/tau/lpath/ldown/wavenumber/index/status .npy。
设计依据见 ql-atmoforge/ModModel.md。
"""

__version__ = "0.1.0"

NETS = ("tau", "lpath", "ldown")
