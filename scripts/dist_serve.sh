#!/bin/bash

# VLLM 分布式部署脚本
# 支持多节点部署、nginx负载均衡、GPU分配

set -e

# 必需参数（用户必须设置）
MODEL_PATH=""
SERVED_MODEL_NAME=""

# 可选参数（仅在用户设置时才传递给vllm）
HOST="0.0.0.0"
DTYPE=""
CHAT_TEMPLATE=""
LIMIT_MM_PER_PROMPT=""
ENABLE_PREFIX_CACHING=""
ENABLE_AUTO_TOOL_CHOICE=""
TOOL_CALL_PARSER=""

# 部署配置
TOTAL_GPUS=8
GPUS_PER_NODE=4
BASE_PORT=8000
NGINX_PORT=8888
SERVER_NAME="127.0.0.1"

# nginx配置文件路径
NGINX_CONF="/etc/nginx/conf.d/load_balancer.conf"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# 显示帮助信息
show_help() {
    cat << EOF
VLLM 分布式部署脚本

用法: $0 [选项] [命令]

命令:
    deploy      部署VLLM服务
    stop        停止所有VLLM服务
    restart     重启所有服务
    status      查看服务状态
    cleanup     清理配置和进程

必需选项:
    -m, --model-path PATH          模型路径 (必需)
    --served-model-name NAME       服务模型名称 (必需)

可选选项:
    -g, --total-gpus NUM           总GPU数量 (默认: $TOTAL_GPUS)
    -n, --gpus-per-node NUM        每个节点GPU数量 (默认: $GPUS_PER_NODE)
    -p, --base-port PORT           基础端口 (默认: $BASE_PORT)
    --nginx-port PORT              Nginx端口 (默认: $NGINX_PORT)
    --server-name NAME             服务器名称 (默认: $SERVER_NAME)
    --dtype TYPE                   数据类型 (如果不设置，使用vllm默认值)
    --chat-template PATH           聊天模板路径 (如果不设置，使用vllm默认值)
    --limit-mm-per-prompt LIMIT    多模态限制 (如果不设置，使用vllm默认值)
    --enable-prefix-caching        启用前缀缓存 (如果不设置，使用vllm默认值)
    --enable-auto-tool-choice      启用自动工具选择 (如果不设置，使用vllm默认值)
    --tool-call-parser PARSER      工具调用解析器 (如果不设置，使用vllm默认值)
    -h, --help                     显示此帮助信息

示例:
    $0 -m /path/to/model --served-model-name my_model deploy
    $0 -m /path/to/model --served-model-name my_model -g 8 -n 4 deploy
    $0 -m /path/to/model --served-model-name my_model --dtype bfloat16 deploy
    $0 stop                        # 停止所有服务
    $0 status                      # 查看服务状态
EOF
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--model-path)
                MODEL_PATH="$2"
                shift 2
                ;;
            --served-model-name)
                SERVED_MODEL_NAME="$2"
                shift 2
                ;;
            -g|--total-gpus)
                TOTAL_GPUS="$2"
                shift 2
                ;;
            -n|--gpus-per-node)
                GPUS_PER_NODE="$2"
                shift 2
                ;;
            -p|--base-port)
                BASE_PORT="$2"
                shift 2
                ;;
            --nginx-port)
                NGINX_PORT="$2"
                shift 2
                ;;
            --server-name)
                SERVER_NAME="$2"
                shift 2
                ;;
            --dtype)
                DTYPE="$2"
                shift 2
                ;;
            --chat-template)
                CHAT_TEMPLATE="$2"
                shift 2
                ;;
            --limit-mm-per-prompt)
                LIMIT_MM_PER_PROMPT="$2"
                shift 2
                ;;
            --enable-prefix-caching)
                ENABLE_PREFIX_CACHING="true"
                shift
                ;;
            --enable-auto-tool-choice)
                ENABLE_AUTO_TOOL_CHOICE="true"
                shift
                ;;
            --tool-call-parser)
                TOOL_CALL_PARSER="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            deploy|stop|restart|status|cleanup)
                COMMAND="$1"
                shift
                ;;
            *)
                error "未知选项: $1"
                echo "使用 -h 或 --help 查看帮助"
                exit 1
                ;;
        esac
    done
}

# 验证必需参数
validate_required_params() {
    if [[ -z "$MODEL_PATH" ]]; then
        error "缺少必需参数: --model-path"
        echo "使用 -h 或 --help 查看帮助"
        return 1
    fi
    
    if [[ -z "$SERVED_MODEL_NAME" ]]; then
        error "缺少必需参数: --served-model-name"
        echo "使用 -h 或 --help 查看帮助"
        return 1
    fi
    
    if [[ ! -d "$MODEL_PATH" ]]; then
        error "模型路径不存在: $MODEL_PATH"
        return 1
    fi
    
    return 0
}

# 检查并安装nginx
install_nginx() {
    log "检查nginx安装状态..."
    
    if command -v nginx >/dev/null 2>&1; then
        log "Nginx已安装"
        return 0
    fi
    
    log "Nginx未安装，开始安装..."
    
    if ! sudo apt update; then
        error "无法更新包列表"
        return 1
    fi
    
    if ! sudo apt install -y nginx; then
        error "Nginx安装失败"
        return 1
    fi
    
    log "Nginx安装成功"
    return 0
}

# 生成nginx配置
generate_nginx_config() {
    local num_nodes=$1
    
    log "生成nginx负载均衡配置..."
    
    # 创建配置文件内容
    cat > /tmp/load_balancer.conf << EOF
upstream model_servers {
    least_conn;
EOF

    # 添加所有节点的服务器配置
    for ((k=0; k<num_nodes; k++)); do
        local port=$((BASE_PORT + k))
        echo "    server $HOST:$port max_fails=3 fail_timeout=30s;" >> /tmp/load_balancer.conf
    done
    
    cat >> /tmp/load_balancer.conf << EOF
    
    keepalive 32;
}

server {
    listen $NGINX_PORT;
    server_name $SERVER_NAME;

    # 增加缓冲区大小
    proxy_buffer_size 128k;
    proxy_buffers 4 256k;
    proxy_busy_buffers_size 256k;
    
    proxy_buffering off;
    
    client_max_body_size 100M;
    
    location /v1 {
        proxy_pass http://model_servers;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;
        send_timeout 600;
        
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
    
    location /health {
        access_log off;
        return 200 "healthy\n";
    }
}
EOF
    
    # 复制配置文件到nginx目录
    if ! sudo cp /tmp/load_balancer.conf "$NGINX_CONF"; then
        error "无法复制nginx配置文件"
        return 1
    fi
    
    rm /tmp/load_balancer.conf
    log "Nginx配置文件已生成: $NGINX_CONF"
    return 0
}

# 生成只包含运行中节点的nginx配置
generate_nginx_config_for_running_nodes() {
    log "生成nginx负载均衡配置（仅包含运行中的节点）..."
    
    # 检查哪些节点正在运行
    local running_nodes=()
    if [[ -d "logs" ]]; then
        for pid_file in logs/vllm_node_*.pid; do
            if [[ -f "$pid_file" ]]; then
                local node_id=$(basename "$pid_file" .pid | sed 's/vllm_node_//')
                local pid=$(cat "$pid_file")
                local port=$((BASE_PORT + node_id))
                
                # 检查进程是否存在且健康
                if kill -0 "$pid" 2>/dev/null && curl -s "http://$HOST:$port/health" >/dev/null 2>&1; then
                    running_nodes+=($node_id)
                fi
            fi
        done
    fi
    
    if [[ ${#running_nodes[@]} -eq 0 ]]; then
        error "没有找到运行中的节点"
        return 1
    fi
    
    log "找到 ${#running_nodes[@]} 个运行中的节点: ${running_nodes[*]}"
    
    # 创建配置文件内容
    cat > /tmp/load_balancer.conf << EOF
upstream model_servers {
    least_conn;
EOF

    # 添加运行中节点的服务器配置
    for node_id in "${running_nodes[@]}"; do
        local port=$((BASE_PORT + node_id))
        echo "    server $HOST:$port max_fails=3 fail_timeout=30s;" >> /tmp/load_balancer.conf
    done
    
    cat >> /tmp/load_balancer.conf << EOF
    
    keepalive 32;
}

server {
    listen $NGINX_PORT;
    server_name $SERVER_NAME;

    # 增加缓冲区大小
    proxy_buffer_size 128k;
    proxy_buffers 4 256k;
    proxy_busy_buffers_size 256k;
    
    proxy_buffering off;
    
    client_max_body_size 100M;
    
    location /v1 {
        proxy_pass http://model_servers;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;
        send_timeout 600;
        
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
    
    location /health {
        access_log off;
        return 200 "healthy\n";
    }
}
EOF
    
    # 复制配置文件到nginx目录
    if ! sudo cp /tmp/load_balancer.conf "$NGINX_CONF"; then
        error "无法复制nginx配置文件"
        return 1
    fi
    
    rm /tmp/load_balancer.conf
    log "Nginx配置文件已生成: $NGINX_CONF (包含 ${#running_nodes[@]} 个节点)"
    return 0
}

# 重启nginx服务
restart_nginx() {
    log "重启nginx服务..."
    
    # 测试配置文件
    if ! sudo nginx -t; then
        error "Nginx配置文件测试失败"
        return 1
    fi
    
    # 重启nginx
    if ! sudo systemctl restart nginx; then
        error "Nginx重启失败"
        sudo systemctl status nginx
        return 1
    fi
    
    # 检查服务状态
    if ! sudo systemctl is-active nginx >/dev/null; then
        error "Nginx服务未正常运行"
        sudo systemctl status nginx
        return 1
    fi
    
    log "Nginx重启成功"
    return 0
}

# 启动VLLM节点
start_vllm_node() {
    local node_id=$1
    local port=$2
    local gpu_start=$3
    local gpu_count=$4
    
    log "启动VLLM节点 $node_id (端口: $port, GPU: $gpu_start-$((gpu_start+gpu_count-1)))"
    
    # 设置CUDA设备
    local cuda_devices=""
    for ((j=0; j<gpu_count; j++)); do
        if [[ $j -gt 0 ]]; then
            cuda_devices+=","
        fi
        cuda_devices+=$((gpu_start + j))
    done
    
    # 创建日志目录
    mkdir -p logs
    
    # 构建vllm命令 - 只添加用户明确设置的参数
    local vllm_cmd="vllm serve"
    vllm_cmd+=" \"$MODEL_PATH\""
    vllm_cmd+=" --port $port"
    vllm_cmd+=" --host $HOST"
    vllm_cmd+=" -tp $gpu_count"
    vllm_cmd+=" --served-model-name $SERVED_MODEL_NAME"
    
    # 只在用户设置时才添加可选参数
    if [[ -n "$DTYPE" ]]; then
        vllm_cmd+=" --dtype $DTYPE"
    fi
    
    if [[ -n "$LIMIT_MM_PER_PROMPT" ]]; then
        vllm_cmd+=" --limit-mm-per-prompt $LIMIT_MM_PER_PROMPT"
    fi
    
    if [[ "$ENABLE_PREFIX_CACHING" == "true" ]]; then
        vllm_cmd+=" --enable-prefix-caching"
    fi
    
    if [[ -n "$CHAT_TEMPLATE" && -f "$CHAT_TEMPLATE" ]]; then
        vllm_cmd+=" --chat-template \"$CHAT_TEMPLATE\""
    fi
    
    if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "true" ]]; then
        vllm_cmd+=" --enable-auto-tool-choice"
    fi
    
    if [[ -n "$TOOL_CALL_PARSER" ]]; then
        vllm_cmd+=" --tool-call-parser $TOOL_CALL_PARSER"
    fi
    
    # 启动服务（后台运行）
    local log_file="logs/vllm_node_${node_id}.log"
    local pid_file="logs/vllm_node_${node_id}.pid"
    
    # 正确设置环境变量并启动服务 - 在子进程中设置环境变量
    CUDA_VISIBLE_DEVICES="$cuda_devices" nohup bash -c "$vllm_cmd" > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    
    local pid=$(cat $pid_file)
    log "VLLM节点 $node_id 已启动 (PID: $pid, CUDA_VISIBLE_DEVICES=$cuda_devices)"
    log "启动命令: CUDA_VISIBLE_DEVICES=$cuda_devices $vllm_cmd"
    
    # 等待服务启动
    log "等待节点 $node_id 启动..."
    local max_wait=180
    local count=0
    while [[ $count -lt $max_wait ]]; do
        if curl -s "http://$HOST:$port/health" >/dev/null 2>&1; then
            log "节点 $node_id 启动成功"
            return 0
        fi
        sleep 2
        count=$((count + 2))
    done
    
    error "节点 $node_id 启动超时"
    log "节点 $node_id 的日志信息:"
    if [[ -f "$log_file" ]]; then
        tail -n 20 "$log_file"
    fi
    return 1
}

# 部署所有节点
deploy_nodes() {
    # 验证必需参数
    if ! validate_required_params; then
        return 1
    fi
    
    local num_nodes=$((TOTAL_GPUS / GPUS_PER_NODE))
    
    if [[ $((TOTAL_GPUS % GPUS_PER_NODE)) -ne 0 ]]; then
        error "GPU数量不能被每节点GPU数量整除: $TOTAL_GPUS / $GPUS_PER_NODE"
        return 1
    fi
    
    log "开始部署 $num_nodes 个节点，每个节点 $GPUS_PER_NODE 张GPU"
    log "模型路径: $MODEL_PATH"
    log "服务模型名称: $SERVED_MODEL_NAME"
    
    # 调试信息
    log "调试信息 - TOTAL_GPUS: $TOTAL_GPUS, GPUS_PER_NODE: $GPUS_PER_NODE, num_nodes: $num_nodes"
    log "调试信息 - BASE_PORT: $BASE_PORT"
    
    # 启动所有VLLM节点 - 改进错误处理，允许部分节点失败
    local failed_nodes=()
    local successful_nodes=0
    
    log "开始启动节点循环，将启动 $num_nodes 个节点"
    for ((i=0; i<num_nodes; i++)); do
        log "循环迭代 $i: 准备启动节点 $i"
        local port=$((BASE_PORT + i))
        local gpu_start=$((i * GPUS_PER_NODE))
        
        log "节点 $i 配置: port=$port, gpu_start=$gpu_start, gpu_count=$GPUS_PER_NODE"
        
        if start_vllm_node $i $port $gpu_start $GPUS_PER_NODE; then
            successful_nodes=$((successful_nodes + 1))
            log "节点 $i 启动成功，successful_nodes=$successful_nodes"
        else
            error "节点 $i 启动失败，继续启动其他节点..."
            failed_nodes+=($i)
            log "节点 $i 启动失败，failed_nodes: ${failed_nodes[*]}"
        fi
        
        log "完成节点 $i 的处理，继续下一个节点"
    done
    
    log "循环结束，共启动 $successful_nodes 个节点"
    
    # 检查是否有成功启动的节点
    if [[ $successful_nodes -eq 0 ]]; then
        error "所有节点启动失败，部署终止"
        return 1
    fi
    
    log "成功启动 $successful_nodes/$num_nodes 个节点"
    if [[ ${#failed_nodes[@]} -gt 0 ]]; then
        log "失败的节点: ${failed_nodes[*]}"
    fi
    
    # 配置nginx - 只为成功启动的节点配置
    if ! install_nginx; then
        return 1
    fi
    
    if ! generate_nginx_config_for_running_nodes; then
        return 1
    fi
    
    if ! restart_nginx; then
        return 1
    fi
    
    log "所有服务部署完成！"
    log "负载均衡器地址: http://$SERVER_NAME:$NGINX_PORT"
    log "成功启动的节点数: $successful_nodes"
    
    return 0
}

# 停止所有服务
stop_services() {
    log "停止所有VLLM服务..."
    
    # 停止VLLM进程
    if [[ -d "logs" ]]; then
        for pid_file in logs/vllm_node_*.pid; do
            if [[ -f "$pid_file" ]]; then
                local pid=$(cat "$pid_file")
                if kill -0 "$pid" 2>/dev/null; then
                    log "停止进程 $pid"
                    kill "$pid"
                fi
                rm -f "$pid_file"
            fi
        done
    fi
    
    # 强制杀死残留进程
    pkill -f "vllm serve" || true
    
    log "所有VLLM服务已停止"
}

# 查看服务状态
show_status() {
    log "检查服务状态..."
    
    # 检查nginx状态
    echo "=== Nginx状态 ==="
    if sudo systemctl is-active nginx >/dev/null; then
        echo "✓ Nginx正在运行"
        echo "  监听端口: $NGINX_PORT"
    else
        echo "✗ Nginx未运行"
    fi
    echo
    
    # 检查VLLM节点状态
    echo "=== VLLM节点状态 ==="
    if [[ -d "logs" ]]; then
        local running_count=0
        for pid_file in logs/vllm_node_*.pid; do
            if [[ -f "$pid_file" ]]; then
                local node_id=$(basename "$pid_file" .pid | sed 's/vllm_node_//')
                local pid=$(cat "$pid_file")
                local port=$((BASE_PORT + node_id))
                
                if kill -0 "$pid" 2>/dev/null; then
                    if curl -s "http://$HOST:$port/health" >/dev/null 2>&1; then
                        echo "✓ 节点 $node_id (PID: $pid, 端口: $port) - 健康"
                        running_count=$((running_count + 1))
                    else
                        echo "⚠ 节点 $node_id (PID: $pid, 端口: $port) - 进程存在但不响应"
                    fi
                else
                    echo "✗ 节点 $node_id - 进程不存在"
                fi
            fi
        done
        echo "运行中的节点: $running_count"
    else
        echo "未找到节点信息"
    fi
}

# 清理配置和进程
cleanup() {
    log "清理配置和进程..."
    
    stop_services
    
    # 删除nginx配置
    if [[ -f "$NGINX_CONF" ]]; then
        sudo rm -f "$NGINX_CONF"
        log "已删除nginx配置文件"
    fi
    
    # 重启nginx以应用配置更改
    if command -v nginx >/dev/null 2>&1; then
        sudo systemctl restart nginx || true
    fi
    
    # 清理日志文件
    if [[ -d "logs" ]]; then
        rm -rf logs
        log "已清理日志文件"
    fi
    
    log "清理完成"
}

# 主函数
main() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 1
    fi
    
    parse_args "$@"
    
    case "$COMMAND" in
        deploy)
            deploy_nodes
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 2
            deploy_nodes
            ;;
        status)
            show_status
            ;;
        cleanup)
            cleanup
            ;;
        *)
            error "未知命令: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
