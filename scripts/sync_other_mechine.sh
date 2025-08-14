#!/bin/bash

# 源目录
# /data1/saves/qwen2_5vl-7b/full/sft_mix_llava_onevision_robospatial_pointing_morewb_1e-6_100k/checkpoint-34000
SRC_DIR="/root/.cursor-server"
# SRC_DIR="/jfs/jensen/code/LLaMA-Factory/data/data/open-x/train/train_datas_all_hf.json"

# 目标机器列表
TARGETS=(
    "unitree-14"
    "unitree-15"
    "unitree-16"
    "unitree-17"
    "unitree-12"
    "unitree-7"
    "h200"
)

# 同步函数
sync_to_target() {
    local target=$1
    echo "正在同步到 $target..."
    
    # 使用rsync同步文件夹，保持权限和时间戳，并自动创建目标目录
    rsync -avz --progress --mkpath "$SRC_DIR/" "$target:$SRC_DIR/"
    
    if [ $? -eq 0 ]; then
        echo "✓ 成功同步到 $target"
    else
        echo "✗ 同步到 $target 失败"
    fi
}

# 主执行逻辑
echo "开始同步文件夹: $SRC_DIR"
echo "目标机器: ${TARGETS[*]}"
echo "-----------------------------------"

# 遍历目标机器进行同步
for target in "${TARGETS[@]}"; do
    sync_to_target "$target"
    echo ""
done

echo "同步完成!"