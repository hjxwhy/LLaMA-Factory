#!/bin/bash

# Multi-node training script with dynamic node configuration
# Supports any number of machines and auto-detects master node

# Default configuration
DEFAULT_MASTER_PORT="29500"
DEFAULT_NPROC_PER_NODE=8
DEFAULT_TRAIN_SCRIPT="src/train.py"
DEFAULT_CONFIG_FILE="examples/train_full/qwen2_5vl_full_sft_72b.yaml"
DEFAULT_CONDA_ENV="jensen_vlm"

# Machine IP mapping (add more machines as needed)
declare -A MACHINE_IP_MAP=(
    ["unitree-7"]="10.3.1.50"
    ["unitree-12"]="10.3.1.212"
    ["unitree-14"]="10.3.1.197"
    ["unitree-15"]="10.3.1.183"
    ["unitree-16"]="10.3.1.51"
    ["unitree-17"]="10.3.1.136"
    # Add more machines here as needed
    # ["machine-name"]="ip-address"
)

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [MACHINES...]"
    echo ""
    echo "Options:"
    echo "  -h, --help                  Show this help message"
    echo "  -p, --port PORT             Master port (default: $DEFAULT_MASTER_PORT)"
    echo "  -g, --gpus GPUS_PER_NODE    Number of GPUs per node (default: $DEFAULT_NPROC_PER_NODE)"
    echo "  -s, --script SCRIPT         Training script path (default: $DEFAULT_TRAIN_SCRIPT)"
    echo "  -c, --config CONFIG         Config file path (default: $DEFAULT_CONFIG_FILE)"
    echo "  -e, --env ENV               Conda environment (default: $DEFAULT_CONDA_ENV)"
    echo "  -m, --master MASTER         Force master node (default: auto-detect current machine)"
    echo "  stop                        Stop training on all configured nodes"
    echo ""
    echo "MACHINES: Space-separated list of machine names (default: unitree-14 unitree-15 unitree-16 unitree-17)"
    echo ""
    echo "Examples:"
    echo "  $0                          # Use default 4 machines, current machine as master"
    echo "  $0 unitree-14 unitree-15    # Use only 2 machines"
    echo "  $0 -p 29501 -g 4 unitree-14 unitree-15 unitree-16  # Custom port and GPUs"
    echo "  $0 stop                     # Stop training"
    echo ""
}

# Function to get IP address for a machine
get_machine_ip() {
    local machine_name=$1
    echo "${MACHINE_IP_MAP[$machine_name]}"
}

# Function to get current machine name
get_current_machine() {
    local full_hostname=$(hostname -s)
    # Remove -GM403 suffix if present to match machine mapping
    echo "$full_hostname" | sed 's/-GM403//'
}

# Function to get current machine IP
get_current_machine_ip() {
    local current_machine=$(get_current_machine)
    get_machine_ip "$current_machine"
}

# Parse command line arguments
MASTER_PORT="$DEFAULT_MASTER_PORT"
NPROC_PER_NODE="$DEFAULT_NPROC_PER_NODE"
TRAIN_SCRIPT="$DEFAULT_TRAIN_SCRIPT"
CONFIG_FILE="$DEFAULT_CONFIG_FILE"
CONDA_ENV="$DEFAULT_CONDA_ENV"
FORCE_MASTER=""
MACHINES=()
STOP_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -p|--port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -g|--gpus)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        -s|--script)
            TRAIN_SCRIPT="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -e|--env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -m|--master)
            FORCE_MASTER="$2"
            shift 2
            ;;
        stop)
            STOP_MODE=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            MACHINES+=("$1")
            shift
            ;;
    esac
done

# Set default machines if none specified
if [ ${#MACHINES[@]} -eq 0 ]; then
    MACHINES=("unitree-14" "unitree-15" "unitree-16" "unitree-17")
fi

# Validate machines and get their IPs
VALID_MACHINES=()
for machine in "${MACHINES[@]}"; do
    ip=$(get_machine_ip "$machine")
    if [ -z "$ip" ]; then
        echo "Error: Unknown machine '$machine'. Please add it to MACHINE_IP_MAP."
        exit 1
    fi
    VALID_MACHINES+=("$machine")
done

# Determine master node
CURRENT_MACHINE=$(get_current_machine)
if [ -n "$FORCE_MASTER" ]; then
    MASTER_MACHINE="$FORCE_MASTER"
    MASTER_ADDR=$(get_machine_ip "$MASTER_MACHINE")
    if [ -z "$MASTER_ADDR" ]; then
        echo "Error: Forced master machine '$FORCE_MASTER' not found in machine IP map."
        exit 1
    fi
elif [[ " ${VALID_MACHINES[@]} " =~ " ${CURRENT_MACHINE} " ]]; then
    # Current machine is in the list, use it as master
    MASTER_MACHINE="$CURRENT_MACHINE"
    MASTER_ADDR=$(get_current_machine_ip)
else
    # Current machine not in list, use first machine as master
    MASTER_MACHINE="${VALID_MACHINES[0]}"
    MASTER_ADDR=$(get_machine_ip "$MASTER_MACHINE")
fi

# Build NODES array with master first (rank 0)
NODES=()
NNODES=${#VALID_MACHINES[@]}
rank=0

# Add master node first
for machine in "${VALID_MACHINES[@]}"; do
    if [ "$machine" == "$MASTER_MACHINE" ]; then
        ip=$(get_machine_ip "$machine")
        NODES+=("$machine:$ip:0")
        break
    fi
done

# Add other nodes
rank=1
for machine in "${VALID_MACHINES[@]}"; do
    if [ "$machine" != "$MASTER_MACHINE" ]; then
        ip=$(get_machine_ip "$machine")
        NODES+=("$machine:$ip:$rank")
        ((rank++))
    fi
done

# Get the current working directory (should be LLaMA-Factory root)
WORK_DIR=$(pwd)

# Function to kill existing training processes on all nodes
cleanup_nodes() {
    echo "Cleaning up existing training processes on all nodes..."
    for node_info in "${NODES[@]}"; do
        IFS=':' read -r hostname ip rank <<< "$node_info"
        if [ "$hostname" == "$CURRENT_MACHINE" ] || [ "$ip" == "$(get_current_machine_ip)" ]; then
            # Local cleanup for current machine
            echo "Cleaning up $hostname (local)..."
            pkill -f 'torchrun.*train.py' || true
        else
            # Remote cleanup for other nodes
            echo "Cleaning up $hostname ($ip)..."
            ssh -o StrictHostKeyChecking=no root@$ip "pkill -f 'torchrun.*train.py' || true"
        fi
    done
    echo "Cleanup completed."
    echo ""
}

# Handle stop command first
if [ "$STOP_MODE" == true ]; then
    echo "Stopping training on all nodes..."
    cleanup_nodes
    echo "Training stopped on all nodes."
    exit 0
fi

# Display configuration
echo "Starting multi-node training..."
echo "Current machine: $CURRENT_MACHINE"
echo "Master: $MASTER_MACHINE ($MASTER_ADDR:$MASTER_PORT)"
echo "Total nodes: $NNODES"
echo "GPUs per node: $NPROC_PER_NODE"
echo "Working directory: $WORK_DIR"
echo "Conda environment: $CONDA_ENV"
echo "Training script: $TRAIN_SCRIPT"
echo "Config file: $CONFIG_FILE"
echo ""
echo "Node configuration:"
for node_info in "${NODES[@]}"; do
    IFS=':' read -r hostname ip rank <<< "$node_info"
    if [ "$hostname" == "$CURRENT_MACHINE" ]; then
        echo "  Rank $rank: $hostname ($ip) - LOCAL"
    else
        echo "  Rank $rank: $hostname ($ip)"
    fi
done
echo ""

# Function to start training on a specific node
start_training_on_node() {
    local hostname=$1
    local ip=$2
    local node_rank=$3
    
    echo "Starting training on $hostname ($ip) with rank $node_rank..."
    
    if [ "$hostname" == "$CURRENT_MACHINE" ] || [ "$ip" == "$(get_current_machine_ip)" ]; then
        # Local execution for current machine
        echo "Starting training on $hostname (local) with rank $node_rank..."
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export NCCL_SOCKET_IFNAME=eth0
        export NCCL_DEBUG=INFO
        export NCCL_ASYNC_ERROR_HANDLING=1
        export NCCL_TIMEOUT=86400000

        nohup torchrun \
            --nnodes=$NNODES \
            --node_rank=$node_rank \
            --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $TRAIN_SCRIPT \
            $CONFIG_FILE \
            > /tmp/training_node_${node_rank}.log 2>&1 &
        echo "Training started on $hostname (rank $node_rank) - local"
    else
        # SSH command to start training on remote nodes
        ssh -o StrictHostKeyChecking=no root@$ip "
            source /jfs/anaconda3/bin/activate
            conda activate $CONDA_ENV
            cd $WORK_DIR
            export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
            export NCCL_SOCKET_IFNAME=eth0
            export NCCL_DEBUG=INFO
            export NCCL_ASYNC_ERROR_HANDLING=1
            export NCCL_TIMEOUT=86400000  

            nohup torchrun \
                --nnodes=$NNODES \
                --node_rank=$node_rank \
                --nproc_per_node=$NPROC_PER_NODE \
                --master_addr=$MASTER_ADDR \
                --master_port=$MASTER_PORT \
                $TRAIN_SCRIPT \
                $CONFIG_FILE \
                > /tmp/training_node_${node_rank}.log 2>&1 &
            echo 'Training started on $hostname (rank $node_rank)'
        " &
    fi
}

# Cleanup existing processes
cleanup_nodes

# Start training on all nodes
echo "Starting training on all nodes..."
for node_info in "${NODES[@]}"; do
    IFS=':' read -r hostname ip rank <<< "$node_info"
    start_training_on_node $hostname $ip $rank
    sleep 2  # Small delay between node startups
done

echo ""
echo "All nodes started. Training is running in background."
echo ""
echo "To monitor training progress:"
echo "  - Check logs on master node ($MASTER_MACHINE): tail -f /tmp/training_node_0.log"
echo "  - Check logs on worker nodes:"
for node_info in "${NODES[@]}"; do
    IFS=':' read -r hostname ip rank <<< "$node_info"
    if [ "$rank" != "0" ]; then
        if [ "$hostname" == "$CURRENT_MACHINE" ]; then
            echo "    Local $hostname: tail -f /tmp/training_node_${rank}.log"
        else
            echo "    $hostname: ssh root@$ip 'tail -f /tmp/training_node_${rank}.log'"
        fi
    fi
done
echo "  - Monitor GPU usage: nvidia-smi (local) or ssh root@<node_ip> 'nvidia-smi'"
echo ""
echo "To stop training on all nodes:"
echo "  bash $0 stop"
echo ""

# Wait for all background SSH processes to complete
wait

echo "Multi-node training setup completed!"