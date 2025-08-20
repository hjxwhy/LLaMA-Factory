#!/bin/bash

# 设置参数
TRAINING_SCRIPT="src/train.py"  # 替换为你的训练脚本
SCRIPT_ARGS="examples/train_full/qwen2_5vl_full_sft.yaml"  # 替换为你的脚本参数
CHECK_INTERVAL=300  # 检查间隔(秒)，例如5分钟
MIN_MEMORY_FREE=130000  # 最小可用显存(MB)，根据你的需求调整

echo "等待GPU资源可用..."

while true; do
    # 获取GPU内存信息
    GPU_INFO=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    
    # 检查是否有足够的显存
    ENOUGH_MEMORY=1
    while IFS= read -r line; do
        if (( line >= MIN_MEMORY_FREE )); then
            echo "发现可用GPU，空闲显存: ${line}MB"
            ENOUGH_MEMORY=0
            break
        fi
    done <<< "$GPU_INFO"
    
    # 如果找到足够显存，启动训练
    if (( ENOUGH_MEMORY == 0 )); then
        echo "启动训练脚本..."
        torchrun --nproc_per_node=8  --nnodes=1 "$TRAINING_SCRIPT" $SCRIPT_ARGS
        echo "训练完成."
        break
    else
        echo "GPU资源不足，等待检查..."
        sleep $CHECK_INTERVAL
    fi
done