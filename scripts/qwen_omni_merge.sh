#! /bin/bash
BASE_MODEL_PATH=/home/unitree/remote_jensen/Qwen2.5-Omni-7B

LORA_CHECKPOINT_PATH=$1
SAVE_PATH=$2
# python scripts/qwen_omni_merge.py merge_lora \
#     --base_model_path $BASE_MODEL_PATH \
#     --lora_checkpoint_path $LORA_CHECKPOINT_PATH \
#     --save_path $SAVE_PATH

python scripts/qwen_omni_merge.py save_full \
    --base_model_path $BASE_MODEL_PATH \
    --saved_thinker_path $LORA_CHECKPOINT_PATH \
    --save_path $SAVE_PATH