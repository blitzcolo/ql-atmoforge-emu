#!/usr/bin/env bash
# 10 个生产数据集（5 波段 × ground/slant）× 2–3 网 全量训练流水线。
#
# 假设：
#   $DATA_ROOT/<band>_<path>_v1        —— ql-atmoforge merge 产出
#   $DATA_ROOT/<band>_<path>_v1_test   —— 独立 random 采样测试集（可选但强烈建议，
#                                          见 ModModel.md §4.2 第 3 步）
# 用法：
#   DATA_ROOT=/data/atmoforge ./scripts/train_all.sh          # 单卡
#   DATA_ROOT=/data/atmoforge NPROC=4 ./scripts/train_all.sh  # 4 卡 DDP
#   OPT=muon ./scripts/train_all.sh                            # 换 Muon 优化器
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT=${DATA_ROOT:-out}
NPROC=${NPROC:-1}
OPT=${OPT:-adamw}
PY=${PYTHON:-python3}

run_train() {
    if [ "$NPROC" -gt 1 ]; then
        torchrun --standalone --nproc_per_node="$NPROC" scripts/train.py "$@"
    else
        $PY scripts/train.py "$@"
    fi
}

# 结构按 ModModel.md §7.1：width 256–512 × blocks 4–6，vis 用 PCA 初始化解码头
declare -A WIDTH=(  [lwir]=256 [mwir]=256 [nir]=384 [swir]=384 [vis]=512 )
declare -A BLOCKS=( [lwir]=4   [mwir]=4   [nir]=4   [swir]=4   [vis]=6   )

for band in lwir mwir nir swir vis; do
  for ptype in ground slant; do
    ds="$DATA_ROOT/${band}_${ptype}_v1"
    [ -d "$ds" ] || { echo "[skip] $ds 不存在"; continue; }
    name=$(basename "$ds")

    test_args=()
    [ -d "${ds}_test" ] && test_args=(--test-dir "${ds}_test")
    pca_args=()
    pca_mode=none
    if [ "$band" = vis ]; then
        pca_args=(--pca tau=150 lpath=100)   # ModModel.md §6.3：δ 100–200、辐亮度 50–100
        pca_mode=head
    fi

    echo "=== [$name] prepare ==="
    $PY scripts/prepare_data.py --data-dir "$ds" "${test_args[@]}" "${pca_args[@]}"

    nets="tau lpath"
    [ "$band" = lwir ] || [ "$band" = mwir ] && nets="tau lpath ldown"
    for net in $nets; do
        echo "=== [$name] train $net ==="
        extra=()
        [ "$pca_mode" = head ] && extra=(--pca-mode head)
        run_train --data-dir "$ds" --net "$net" \
            --width "${WIDTH[$band]}" --blocks "${BLOCKS[$band]}" \
            --optimizer "$OPT" --batch-size 1024 --preload "${extra[@]}"

        if [ -d "${ds}_test" ]; then
            tag="${WIDTH[$band]}x${BLOCKS[$band]}_${OPT}"
            [ "$pca_mode" = head ] && tag="${tag}_pca-head"
            $PY scripts/evaluate.py --run-dir "runs/${name}_${net}_${tag}" \
                --data-dir "${ds}_test"
        fi
    done
  done
done
echo "=== 全部完成 ==="
