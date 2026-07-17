# ql-atmoforge-emu — MODTRAN 代理网络训练端

配套 [ql-atmoforge](https://github.com/blitzcolo/ql-atmoforge) 数据生成器的 PyTorch 训练侧项目：
把 `ql-atmoforge merge` 产出的数据集训练成 **(大气/几何/太阳参数 → 大气光谱量)**
的神经网络代理模型。

```
ql-atmoforge gen/merge  →  prepare_data.py (×2 prep)  →  train.py (×3 网)  →  evaluate.py
   (C++, Windows 侧)        清洗/划分/统计，见 §3         单卡或 DDP 多卡        独立测试集/锚点
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

# 1) 双 prep（推荐，理由见 §3）：
#    prep_sat  —— 剔整谱饱和样本，给 tau 网
#    prep_full —— 全量保留，给辐亮度网（lpath/ldown）
python scripts/prepare_data.py --data-dir out/lwir_ground_v1 \
    --prep-dir out/lwir_ground_v1/prep_sat --nets tau \
    --test-dir out/lwir_ground_v1_test        # 独立 random 测试集，强烈建议
python scripts/prepare_data.py --data-dir out/lwir_ground_v1 \
    --prep-dir out/lwir_ground_v1/prep_full --nets lpath ldown \
    --saturation-tau-max 0 \
    --test-dir out/lwir_ground_v1_test
#    以上两条等价于：bash scripts/prepare_all.sh lwir_ground_v1
#    全部数据集批量双 prep：DATA_ROOT=out bash scripts/prepare_all.sh

# 2) 训练三个网（thermal 波段 tau/lpath/ldown；反射波段只有 tau/lpath）
python scripts/train.py --data-dir out/lwir_ground_v1 --net tau   \
    --prep-dir out/lwir_ground_v1/prep_sat  --preload
python scripts/train.py --data-dir out/lwir_ground_v1 --net lpath \
    --prep-dir out/lwir_ground_v1/prep_full --preload
python scripts/train.py --data-dir out/lwir_ground_v1 --net ldown \
    --prep-dir out/lwir_ground_v1/prep_full --preload

# 3) 独立测试集评估（NRMSE / τ 绝对误差 / 亮温误差，含 P99）。
#    评辐亮度网时传 --saturation-tau-max 0，让雨雾区进指标（与训练口径一致）
python scripts/evaluate.py --run-dir runs/lwir_ground_v1_tau_256x4_adamw \
    --data-dir out/lwir_ground_v1_test
python scripts/evaluate.py --run-dir runs/lwir_ground_v1_lpath_256x4_adamw \
    --data-dir out/lwir_ground_v1_test --saturation-tau-max 0
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

PCA 基随各自的 prep 拟合（tau 的基在 prep_sat，lpath 的基在 prep_full）：

```sh
python scripts/prepare_data.py --data-dir out/vis_ground_v1 \
    --prep-dir out/vis_ground_v1/prep_sat --nets tau --pca tau=150 \
    --test-dir out/vis_ground_v1_test
python scripts/prepare_data.py --data-dir out/vis_ground_v1 \
    --prep-dir out/vis_ground_v1/prep_full --nets lpath --pca lpath=100 \
    --saturation-tau-max 0 --test-dir out/vis_ground_v1_test
python scripts/train.py --data-dir out/vis_ground_v1 --net lpath \
    --prep-dir out/vis_ground_v1/prep_full \
    --config configs/example_vis_lpath.json          # width 512×6, PCA 初始化解码头
```

`--pca-mode head`（推荐）：PCA 基初始化最后一层线性解码头，随整网端到端微调；
`--pca-mode coeff`：完全在系数空间训练（正交基 ⇒ 系数 MSE ≡ 谱 MSE），显存最省。

### 推理 / 部署（scripts/infer.py）

run 目录自包含（`best.pt` 含 EMA 权重 + norm/pca 副本），可整目录拎走。示例脚本
一次加载同一数据集的 2–3 张网，输出物理光谱 npz：

```sh
python scripts/infer.py \
    --run-dir runs/lwir_ground_v1_tau_256x4_muon \
              runs/lwir_ground_v1_lpath_256x4_muon \
              runs/lwir_ground_v1_ldown_256x4_muon \
    --data-dir out/lwir_ground_v1_test --index 0 5 42 --out pred.npz
# 或 --features-json scene.json（{特征名: 值}，布局= manifest["feature_names"]）
```

自定义集成时按 `infer.load_run()` 的流程走即可（ckpt→InputSpec→前向→OutputSpec.inverse）。
**部署契约四条**：① 输入必须落在 manifest 采样域内；② δ 下钳 0 后 `τ=exp(−δ)`，
δ_min > 7 的视线按完全不透明处理（τ=0，像素 = L_path——τ 网在饱和区无监督，见 §3）；
③ 辐亮度输出下钳 0（log 反变换在近零区可能给出 −ε 量级负值）；④ log 通道上钳
2×训练集通道最大值（vmax = log_eps×1e4 可从 norm 文件还原）——exp 逆变换无上界，
暗场景里深吸收带芯会喷出高若干量级的假能量尖峰，摧毁带积分辐亮度（锚点过闸实测）。
③④ 已内建在 `infer.load_run` 的前向闭包里，自定义集成直接复用它即可。
下游合成 `L_sensor = τ·L_target + L_path`，目标项用 ldown 组装；MWIR 夜间把
SOL_SCAT 分量置零。

### 导出 safetensors（scripts/export_safetensors.py，C++ 渲染端）

每个 run 目录导出为单个自包含文件 `<band>_<geom>_<net>.safetensors`
（geom：horizontal→ground、slant_to_ground→slant、sky→sky）：EMA 权重 +
逐目标归一化数组（F32 张量，log_mask 为 U8）+ JSON 元数据串（输入 spec、
波段网格、目标定义、出处）。C++ 加载端不再需要 run 目录、manifest 或任何
Python 产物；上节部署契约里的不透明门禁（opaque_delta=7）与 δ 钳 20 也随
元数据携带。

```sh
python scripts/export_safetensors.py --all runs/ --out-dir export/atmos_models
# 或指定单个 run：
python scripts/export_safetensors.py --run-dir runs/lwir_ground_v1_tau_256x4_muon
# 训练结束顺手导出：train.py ... --export-safetensors export/atmos_models
```

每个导出文件默认做自检（`--no-check` 关闭）：只凭导出文件重建独立 numpy
前向（镜像 C++ 算法：特征装配 → fp32 ResMLP → float64 逆变换），在随机域内
参数上与 torch 参考前向比对，max|dz| 与 max|dY|/peak 均须 ≤ 1e-4。
未安装 safetensors 包时使用内置写入器，格式相同，不构成硬依赖。

### 锚点过闸（atmospheres.json 8 预设 × 10 数据集）

```sh
python scripts/make_anchor_configs.py --forge-root /path/to/ql-atmoforge
# ql-atmoforge 侧：对 configs/anchors/*.json 逐一 gen + merge（80 份单样本锚点）
python scripts/anchor_gate.py --anchors-root /path/to/ql-atmoforge/out/anchors
```

锚点 = "网络 vs 直跑 MODTRAN 真值"的端到端验收（独立测试集只能验证同采样口径内
的插值）。n=1 单谱判据见 anchor_gate.py 文档串；整谱饱和 / 暗场景记 SAT 跳过。

## 3. 训练前清洗规则（prepare_data.py 做的事）

| 规则 | 默认 | 依据 |
|---|---|---|
| `status=2`（失败占位，谱全零） | 剔除 | - |
| `status=1`（partial） | 保留；目标列含 NaN 的剔除；`--drop-partial` 全剔 | 同上 |
| 整谱饱和：max_ν(tau.TOTAL) < 1e-3 | **仅 tau 的 prep 剔除**；辐亮度网的 prep 用 `--saturation-tau-max 0` 全量保留（双 prep，见下） | 饱和区 δ 标签是垃圾，L 标签是好的 |
| δ = LOG_TOTAL 上钳 20 | 训练目标层面钳制 | τ<2e-9 已无信息 |
| 划分 | val 10% 同集切分 + 测试集用 `--test-dir` 独立 random 数据集 | Sobol 子段互不独立 |

无 `--test-dir` 时会退化为同集切分并打警告——只可用于冒烟，不可用于正式指标。

### 双 prep 目录：`prep_sat`（tau）+ `prep_full`（lpath/ldown）

整谱饱和样本（浓雨/雾 + 长路径，lwir_ground_v1 实测占 21.6%，87% 来自 icld=6 雨云）
对两类目标的意义完全相反：

- **δ = −ln τ 标签在该区是垃圾**：实测 δ 中位 33、最大 999.9（MODTRAN `-LOG` 列打印
  哨兵值），钳 20 后整条谱变常数假标签；留在训练集里还会把逐通道 σ 抬高约 4 倍，
  透明区（下游唯一在用的区域）的损失权重被稀释一个量级。**必须剔。**
- **辐亮度标签在该区是干净的**：E 格式打印不饱和，L_path/L_down 平滑趋于黑体平台
  B(T)——而这正是浓雨/雾成像时像素值的主导项。**必须留**，否则辐亮度网在 icld=6
  格子里损失过半训练密度，关键工况变成外推区。

所以样本选择策略恰好两种，prep 目录就是两个（选择/划分是 prep 的全局属性，逐网
归一化统计本来就分文件存放在同一 prep 内）。**不需要第三个目录**：lpath 与 ldown
的选择策略永远相同（全量），无论波段；反射波段只有 tau/lpath 两张网，同样双目录。
若未来出现逐网差异（如只对 ldown 剔 partial），`--prep-dir` 支持任意多套。

**下游推理契约**：tau 网只在 δ_min ≤ ~7 的域内有监督；预测 δ_min > 7 时按完全
不透明闭合——τ=0，像素 = L_path（辐亮度网在该区有真实监督，可放心查询）。

## 4. 三网速查（自动按 manifest 配置，无需手写维度）

| 网 | 输入（自动剔除） | 目标 | 备注 |
|---|---|---|---|
| `tau` | 全部 − 太阳 2 维 | `tau.LOG_TOTAL`（δ=−ln τ，钳 20） | τ 与太阳无关 |
| `lpath` | 全部 | `lpath.TOTAL_RAD`；**mwir 拆 `PTH_THRML`+`SOL_SCAT` 两头** | 夜间下游把 SOL_SCAT 置零 |
| `ldown` | 斜程剔 h1/视角/距离；横程只剔距离 | `ldown.TOTAL_RAD` | 仅 thermal 波段；**sky 数据集无此网** |

**sky 数据集**（`<band>_sky_v1`，path_type=sky，望天路径）：`lpath` 网即
**(方向, 天气) → 天空穹顶辐亮度**，输入自动含 `cos_view_zenith`（[cos 89°, 1]）；
无派生 range 特征、无 ldown 块。辐亮度 prep 用 `--drop-sun-cone 3` 剔圆日锥体
（prepare_all.sh 已内建）——部署侧穹顶 LUT 烘焙对 3° 锥内与 >89° 的查询做钳制。

输入变换全部由 manifest 解析生成（uniform→min-max、log_uniform→ln、
天顶角→cos、相对方位角→cos Δφ、离散→one-hot、常量剔除），不碰数据统计量；
输出逐通道 μ/σ 标准化 + 方差下限（防 tape7 四位小数量化噪声放大；下限按 log/线性
通道**分组**取各自中位——混组会让 ln 空间的 O(1) σ 把线性通道整组碾平），
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
│   ├── prepare_data.py      # 清洗 + 划分 + 归一化统计 + PCA（--prep-dir 指定产物目录）
│   ├── prepare_all.sh       # 全部数据集批量双 prep（prep_sat/prep_full，可断点续跑）
│   ├── train.py             # 单卡 / torchrun DDP，bf16 + EMA + cosine + 早停
│   ├── evaluate.py          # 独立测试集 / 锚点集评估；辐亮度相对误差/亮温只在
│   │                        #   有信号元素上统计（rad/bt_valid_frac 是覆盖率）
│   ├── infer.py             # 示例推理：run 目录 → 物理光谱 npz（含不透明门禁
│   │                        #   与辐亮度物理守卫；load_run 可当库用）
│   ├── export_safetensors.py # run 目录 → 自包含 safetensors 包（C++ 渲染端）
│   │                        #   ，含 numpy vs torch 前向一致性自检
│   ├── make_anchor_configs.py # 8 预设 × 10 数据集 → 80 份单样本锚点 config
│   ├── anchor_gate.py       # 锚点过闸：每网 × 8 锚点 vs MODTRAN 真值
│   ├── train_all.sh         # 10 数据集全量流水线
│   └── smoke_test.sh        # CPU 全链路自测
└── configs/                 # train.py --config 示例

<data-dir>/prep_sat/         # prepare 产物（tau 用，剔饱和）：keep_mask、
<data-dir>/prep_full/        #   train/val(/test)_idx、norm_<net>.{json,targets.json,npz}、
                             #   pca_<net>.npz；prep_full 给 lpath/ldown（全量）。
                             #   不指定 --prep-dir 时默认写 <data-dir>/prep/（单 prep 流程）
runs/<dataset>_<net>_<tag>/  # config.json、best.pt/last.pt（含 EMA 权重）、
                             #   metrics.csv、norm/pca 副本（自包含，可直接部署）、
                             #   eval_<testset>.json
export/atmos_models/         # export_safetensors.py 默认产物目录：
                             #   <band>_<geom>_<net>.safetensors（C++ 端直接加载）
```

## 7. 已知约束

- 评估/部署时输入必须落在 manifest 采样范围内（含收窄后的范围），域外要么拒绝
  要么重新生成数据集。
- 扩容只允许加大生成端 `n_samples` 后重新 merge + prepare；seed/sampler/范围不许动。
- `--preload` 把目标谱载入内存：lwir/mwir/nir/swir 均 <1.5 GB 建议开；
  vis 单网约 1.3 GB，云端内存够也建议开，否则走 mmap。
- 训练显存（bf16, batch 1024, width 512×6, K=12501 dense 头）也远小于 8 GB；
  PCA 系数空间训练更小。瓶颈从来不是显存，是生成数据的机时。
