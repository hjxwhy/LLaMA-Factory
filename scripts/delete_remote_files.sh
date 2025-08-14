#!/bin/bash

# 要删除的目录或文件路径
# 注意: 请谨慎设置此路径，删除操作不可逆！
TARGET_PATH="/data1/saves/qwen2_5vl-7b/full/vlm_mix_robot_openx_training_50k"

# 目标机器列表
TARGETS=(
    "unitree-15"
    "unitree-16"
    "unitree-17"
)

# 删除确认标志 (设为true时跳过确认)
SKIP_CONFIRMATION=false

# 是否删除本地文件 (设为true时也删除本地文件)
DELETE_LOCAL=false

# 删除远程目标函数
delete_from_target() {
    local target=$1
    echo "正在从 $target 删除 $TARGET_PATH..."
    
    # 首先检查目标路径是否存在
    ssh "$target" "test -e '$TARGET_PATH'"
    if [ $? -ne 0 ]; then
        echo "⚠ $target 上不存在路径: $TARGET_PATH"
        return 0
    fi
    
    # 执行删除操作
    ssh "$target" "rm -rf '$TARGET_PATH'"
    
    if [ $? -eq 0 ]; then
        echo "✓ 成功从 $target 删除"
    else
        echo "✗ 从 $target 删除失败"
    fi
}

# 删除本地文件函数
delete_local() {
    echo "正在删除本地文件: $TARGET_PATH..."
    
    # 检查本地路径是否存在
    if [ ! -e "$TARGET_PATH" ]; then
        echo "⚠ 本地不存在路径: $TARGET_PATH"
        return 0
    fi
    
    # 执行本地删除操作
    rm -rf "$TARGET_PATH"
    
    if [ $? -eq 0 ]; then
        echo "✓ 成功删除本地文件"
    else
        echo "✗ 删除本地文件失败"
    fi
}

# 确认删除函数
confirm_deletion() {
    if [ "$SKIP_CONFIRMATION" = true ]; then
        return 0
    fi
    
    echo "⚠️  警告: 即将删除以下路径:"
    echo "   路径: $TARGET_PATH"
    echo "   远程机器: ${TARGETS[*]}"
    if [ "$DELETE_LOCAL" = true ]; then
        echo "   本地机器: $(hostname)"
    fi
    echo ""
    echo "此操作不可逆！请确认是否继续？"
    read -p "输入 'YES' 继续删除，其他任意键取消: " confirmation
    
    if [ "$confirmation" = "YES" ]; then
        return 0
    else
        echo "操作已取消"
        exit 0
    fi
}

# 主执行逻辑
echo "删除远程文件脚本"
echo "目标路径: $TARGET_PATH"
echo "远程机器: ${TARGETS[*]}"
if [ "$DELETE_LOCAL" = true ]; then
    echo "本地机器: $(hostname)"
fi
echo "-----------------------------------"

# 确认删除操作
confirm_deletion

echo "开始删除操作..."
echo ""

# 删除本地文件 (如果启用)
if [ "$DELETE_LOCAL" = true ]; then
    delete_local
    echo ""
fi

# 遍历目标机器进行删除
for target in "${TARGETS[@]}"; do
    delete_from_target "$target"
    echo ""
done

echo "删除操作完成!"
echo ""
echo "提示: 如需跳过确认提示，可在脚本中设置 SKIP_CONFIRMATION=true"
echo "提示: 如需禁用本地文件删除，可在脚本中设置 DELETE_LOCAL=false" 