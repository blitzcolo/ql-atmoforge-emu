#!/usr/bin/env bash
# 全链路冒烟测试（CPU 可跑，约 2–5 分钟）：
# 伪数据（斜程 thermal + 横程） → 清洗/划分/归一化/PCA → 三网训练（含 Muon、PCA 头） → 评估
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
TMP=tmp/smoke
rm -rf "$TMP"

echo "=== 1. 伪数据（结构 = ql-atmoforge merge 理论产出） ==="
$PY scripts/make_fake_data.py --out $TMP/fake_mwir_slant --n 2048 --seed 42 --sampler sobol
$PY scripts/make_fake_data.py --out $TMP/fake_mwir_slant_test --n 512 --seed 7 --sampler random
$PY scripts/make_fake_data.py --out $TMP/fake_swir_ground --n 1024 --seed 42 \
    --band swir_toy --path horizontal

echo "=== 2. 清洗 + 划分 + 归一化 + PCA ==="
$PY scripts/prepare_data.py --data-dir $TMP/fake_mwir_slant \
    --test-dir $TMP/fake_mwir_slant_test --pca lpath=24
$PY scripts/prepare_data.py --data-dir $TMP/fake_swir_ground --val-frac 0.15

echo "=== 3. 三网训练（斜程 thermal：tau=AdamW, lpath=Muon, ldown=AdamW） ==="
COMMON="--width 64 --blocks 2 --batch-size 256 --num-workers 0 --preload --patience 99"
$PY scripts/train.py --data-dir $TMP/fake_mwir_slant --net tau   $COMMON \
    --epochs 12 --out $TMP/runs/tau
$PY scripts/train.py --data-dir $TMP/fake_mwir_slant --net lpath $COMMON \
    --epochs 12 --optimizer muon --out $TMP/runs/lpath_muon
$PY scripts/train.py --data-dir $TMP/fake_mwir_slant --net ldown $COMMON \
    --epochs 8 --out $TMP/runs/ldown

echo "=== 4. PCA 初始化解码头（端到端微调） ==="
$PY scripts/train.py --data-dir $TMP/fake_mwir_slant --net lpath $COMMON \
    --epochs 8 --pca-mode head --out $TMP/runs/lpath_pca

echo "=== 5. 横程数据集（几何块走另一条代码路径） ==="
$PY scripts/train.py --data-dir $TMP/fake_swir_ground --net tau $COMMON \
    --epochs 6 --out $TMP/runs/swir_tau

echo "=== 6. 独立 random 测试集评估 ==="
for run in tau lpath_muon ldown lpath_pca; do
    $PY scripts/evaluate.py --run-dir $TMP/runs/$run --data-dir $TMP/fake_mwir_slant_test
done
$PY scripts/evaluate.py --run-dir $TMP/runs/swir_tau --data-dir $TMP/fake_swir_ground

echo "=== SMOKE OK ==="
