#!/bin/bash

model_path=/data1/saves/qwen2_5vl-7b/full/vlm_mix_robot_openx_training_100k/checkpoint-9000
output_dir="output"


task_infer_image="assets/demo/task_infer.png"
text="Move the silver pot from in front of the red can, to next to the blue towel at the front edge of the table."
task="task_infer"

python scripts/qwen2_5_vl_infer.py \
    --model_name_or_path $model_path \
    --image $task_infer_image \
    --text "$text" \
    --output_dir $output_dir \
    --task $task

traj_image="assets/demo/trajectory.jpg"
text="pick up the banana"
task="trajectory"

python scripts/qwen2_5_vl_infer.py \
    --model_name_or_path $model_path \
    --image $traj_image \
    --text "$text" \
    --output_dir $output_dir \
    --task $task

pointing_image="assets/demo/pointing.jpg"
text="the blue cup"
task="pointing"

python scripts/qwen2_5_vl_infer.py \
    --model_name_or_path $model_path \
    --image $pointing_image \
    --text "$text" \
    --output_dir $output_dir \
    --task $task

grounding_image="assets/demo/grounding.jpg"
text="banana"
task="grounding"

python scripts/qwen2_5_vl_infer.py \
    --model_name_or_path $model_path \
    --image $grounding_image \
    --text "$text" \
    --output_dir $output_dir \
    --task $task

affordance_image="assets/demo/affordance.jpg"
text="take up the cup"
task="affordance"

python scripts/qwen2_5_vl_infer.py \
    --model_name_or_path $model_path \
    --image $affordance_image \
    --text "$text" \
    --output_dir $output_dir \
    --task $task

spatial_reasoning_image="assets/demo/pointing.jpg"
text="the empty space between the cups close to right cup"
task="spatial_reasoning"

python scripts/qwen2_5_vl_infer.py \
    --model_name_or_path $model_path \
    --image $spatial_reasoning_image \
    --text "$text" \
    --output_dir $output_dir \
    --task $task

task_planning_image="assets/demo/task_infer.png"
text="Move the silver pot from in front of the red can, to next to the blue towel at the front edge of the table."
task="task_planning"

python scripts/qwen2_5_vl_infer.py \
    --model_name_or_path $model_path \
    --image $task_planning_image \
    --text "$text" \
    --output_dir $output_dir \
    --task $task

