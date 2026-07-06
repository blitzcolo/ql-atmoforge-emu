#!/usr/bin/env bash
# 双 prep 批量预处理（README §3）：对每个数据集生成
#   <ds>/prep_sat  —— 剔整谱饱和样本，--nets tau
#   <ds>/prep_full —— 全量保留（--saturation-tau-max 0），--nets lpath [ldown]
#
# 假设：
#   $DATA_ROOT/<band>_<path>_v1        —— ql-atmoforge merge 产出
#   $DATA_ROOT/<band>_<path>_v1_test   —— 独立 random 采样测试集（可选但强烈建议，
#                                          见 ModModel.md §4.2 第 3 步）
# 用法：
#   DATA_ROOT=/data/atmoforge ./scripts/prepare_all.sh               # 全部 10 个
#   DATA_ROOT=/data/atmoforge ./scripts/prepare_all.sh lwir_ground_v1 vis_slant_v1
#   FORCE=1 DATA_ROOT=... ./scripts/prepare_all.sh                   # 已有 prep 也重跑
#
# 已存在 splits.json 的 prep 目录默认跳过（断点续跑友好）；数据重新 merge 后用 FORCE=1。
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT=${DATA_ROOT:-out}
PY=${PYTHON:-python3}
FORCE=${FORCE:-0}

datasets=("$@")
if [ ${#datasets[@]} -eq 0 ]; then
    for band in lwir mwir nir swir vis; do
        for ptype in ground slant; do datasets+=("${band}_${ptype}_v1"); done
    done
fi

prep_one() {  # $1 = prep 目录，其余 = prepare_data.py 参数
    local dir=$1; shift
    if [ "$FORCE" != 1 ] && [ -f "$dir/splits.json" ]; then
        echo "  [skip] $dir 已存在（FORCE=1 重跑）"
        return 0
    fi
    $PY scripts/prepare_data.py --prep-dir "$dir" "$@"
}

for name in "${datasets[@]}"; do
    ds="$DATA_ROOT/$name"
    [ -d "$ds" ] || { echo "[skip] $ds 不存在"; continue; }
    band=${name%%_*}

    test_args=()
    [ -d "${ds}_test" ] && test_args=(--test-dir "${ds}_test")
    [ ${#test_args[@]} -eq 0 ] && \
        echo "[warn] $name: 无 ${name}_test，退化为同集切分（仅可冒烟，不可出正式指标）"

    rad_nets="lpath"
    { [ "$band" = lwir ] || [ "$band" = mwir ]; } && rad_nets="lpath ldown"
    pca_sat=(); pca_full=()
    if [ "$band" = vis ]; then
        pca_sat=(--pca tau=150); pca_full=(--pca lpath=100)   # ModModel.md §6.3
    fi

    echo "=== [$name] prep_sat (tau, 剔饱和) ==="
    prep_one "$ds/prep_sat" --data-dir "$ds" --nets tau \
        "${test_args[@]}" "${pca_sat[@]}"
    echo "=== [$name] prep_full ($rad_nets, 全量) ==="
    prep_one "$ds/prep_full" --data-dir "$ds" --nets $rad_nets \
        --saturation-tau-max 0 "${test_args[@]}" "${pca_full[@]}"
done
echo "=== prepare_all 完成 ==="
