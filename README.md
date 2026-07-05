# ql-atmoforge-emu — MODTRAN 代理网络训练端

配套 [ql-atmoforge](https://github.com/blitzcolo/ql-atmoforge) 数据生成器的 PyTorch 训练侧项目：
把 `ql-atmoforge merge` 产出的数据集训练成 **(大气/几何/太阳参数 → 大气光谱量)**
的神经网络代理模型。

```
ql-atmoforge gen/merge  →  prepare_data.py  →  train.py (×3 网)  →  evaluate.py
   (C++, Windows 侧)        清洗/划分/统计       单卡或 DDP 多卡        独立测试集/锚点
```

## 1. 环境

Python ≥ 3.10，PyTorch + CUDA 12：(13 则修改相应版本号即可)

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA 12.x
pip install -r requirements.txt
```

本地无 GPU 时装 CPU 轮子即可跑冒烟测试：
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

torch ≥ 2.9 有原生 `torch.optim.Muon`；更老版本 `--optimizer muon`
自动回退到内置 SimpleMuon（Newton–Schulz + RMS 匹配），行为一致。

## 2. 快速开始（单个数据集）

```sh
# 0) 数据还没生成？先跑全链路冒烟（合成数据，CPU 几分钟）
bash scripts/smoke_test.sh

# 1) 清洗 + 自动划分 + 归一化统计（产物写入 <data-dir>/prep/）
python scripts/prepare_data.py --data-dir out/lwir_ground_v1 \
    --test-dir out/lwir_ground_v1_test        # 独立 random 测试集，强烈建议

# 2) 训练三个网（thermal 波段 tau/lpath/ldown；反射波段只有 tau/lpath）
python scripts/train.py --data-dir out/lwir_ground_v1 --net tau   --preload
python scripts/train.py --data-dir out/lwir_ground_v1 --net lpath --preload
python scripts/train.py --data-dir out/lwir_ground_v1 --net ldown --preload

# 3) 独立测试集评估（NRMSE / τ 绝对误差 / 亮温误差，含 P99）
python scripts/evaluate.py --run-dir runs/lwir_ground_v1_tau_256x4_adamw \
    --data-dir out/lwir_ground_v1_test
```

多卡（同机 N 卡，DDP）：

```sh
torchrun --standalone --nproc_per_node=4 scripts/train.py \
    --data-dir out/mwir_slant_v1 --net lpath --batch-size 1024 --preload
```

10 个生产数据集一把梭（含 prepare + 训练 + 评估）：

```sh
DATA_ROOT=/data/atmoforge NPROC=4 bash scripts/train_all.sh
```

### vis 波段（K=12501，必须 PCA）

```sh
python scripts/prepare_data.py --data-dir out/vis_ground_v1 \
    --test-dir out/vis_ground_v1_test --pca tau=150 lpath=100
python scripts/train.py --data-dir out/vis_ground_v1 --net lpath \
    --config configs/example_vis_lpath.json          # width 512×6, PCA 初始化解码头
```

`--pca-mode head`（推荐）：PCA 基初始化最后一层线性解码头，随整网端到端微调；
`--pca-mode coeff`：完全在系数空间训练（正交基 ⇒ 系数 MSE ≡ 谱 MSE），显存最省。

## 3. 训练前清洗规则（prepare_data.py 做的事）

| 规则 | 默认 | 依据 |
|---|---|---|
| `status=2`（失败占位，谱全零） | 剔除 | - |
| `status=1`（partial） | 保留；目标列含 NaN 的剔除；`--drop-partial` 全剔 | 同上 |
| 整谱饱和：max_ν(tau.TOTAL) < 1e-3 | 剔除（`--saturation-tau-max` 可调/关） | 打印下限 0.0000 压平光谱 |
| δ = LOG_TOTAL 上钳 20 | 训练目标层面钳制 | τ<2e-9 已无信息 |
| 划分 | val 10% 同集切分 + 测试集用 `--test-dir` 独立 random 数据集 | Sobol 子段互不独立 |

无 `--test-dir` 时会退化为同集切分并打警告——只可用于冒烟，不可用于正式指标。

## 4. 三网速查（自动按 manifest 配置，无需手写维度）

| 网 | 输入（自动剔除） | 目标 | 备注 |
|---|---|---|---|
| `tau` | 全部 − 太阳 2 维 | `tau.LOG_TOTAL`（δ=−ln τ，钳 20） | τ 与太阳无关 |
| `lpath` | 全部 | `lpath.TOTAL_RAD`；**mwir 拆 `PTH_THRML`+`SOL_SCAT` 两头** | 夜间下游把 SOL_SCAT 置零 |
| `ldown` | 斜程剔 h1/视角/距离；横程只剔距离 | `ldown.TOTAL_RAD` | 仅 thermal 波段 |

输入变换全部由 manifest 解析生成（uniform→min-max、log_uniform→ln、
天顶角→cos、相对方位角→cos Δφ、离散→one-hot、常量剔除），不碰数据统计量；
输出逐通道 μ/σ 标准化 + 方差下限（防 tape7 四位小数量化噪声放大），
动态范围 >10³ 的辐亮度通道自动改 log(L+ε)。所有常数存 `prep/` 与 run 目录随模型走。

## 5. 优化器：AdamW（默认）与 Muon

`--optimizer muon` 使用混合方案：**残差块内的 2D 权重矩阵走 Muon，
输入层/输出头/LayerNorm/bias 仍走 AdamW**（Muon 的标准用法，首末层不做正交化）。
lr/weight_decay 直接沿用 AdamW 的取值（原生 Muon 用 `match_rms_adamw`，
回退实现用 0.2·√max(m,n) RMS 匹配缩放），因此两个优化器可用同一组超参 A/B 对比。

建议：先用 AdamW 跑通基线，再在同一数据集上开 `--optimizer muon` 对比
val 损失曲线——表格型 MLP 基准里 Muon 是少数能稳定小胜 AdamW 的优化器，
但本项目网络很小（1–3M 参数），预期收益是收敛步数而非最终精度上限。

## 6. 目录与产物

```
ql-atmoforge-emu/
├── atmoemu/                 # 库：manifest / transforms / data / model / optim / metrics
├── scripts/
│   ├── make_fake_data.py    # 合成数据（结构 = merge 理论产出），冒烟用
│   ├── prepare_data.py      # 清洗 + 划分 + 归一化统计 + PCA
│   ├── train.py             # 单卡 / torchrun DDP，bf16 + EMA + cosine + 早停
│   ├── evaluate.py          # 独立测试集 / 锚点集评估
│   ├── train_all.sh         # 10 数据集全量流水线
│   └── smoke_test.sh        # CPU 全链路自测
└── configs/                 # train.py --config 示例

<data-dir>/prep/             # prepare 产物：keep_mask、train/val(/test)_idx、
                             #   norm_<net>.{json,targets.json,npz}、pca_<net>.npz
runs/<dataset>_<net>_<tag>/  # config.json、best.pt/last.pt（含 EMA 权重）、
                             #   metrics.csv、norm/pca 副本（自包含，可直接部署）、
                             #   eval_<testset>.json
```

## 7. 已知约束

- 评估/部署时输入必须落在 manifest 采样范围内（含收窄后的范围），域外要么拒绝
  要么重新生成数据集。
- 扩容只允许加大生成端 `n_samples` 后重新 merge + prepare；seed/sampler/范围不许动。
- `--preload` 把目标谱载入内存：lwir/mwir/nir/swir 均 <1.5 GB 建议开；
  vis 单网约 1.3 GB，云端内存够也建议开，否则走 mmap。
- 训练显存（bf16, batch 1024, width 512×6, K=12501 dense 头）也远小于 8 GB；
  PCA 系数空间训练更小。瓶颈从来不是显存，是生成数据的机时。
