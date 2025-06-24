import json
import os
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw
import numpy as np

# Model setup
model_name = "/DATA/disk0/saves/qwen2_5vl-7b/full/sft_mix_llava_onevision_robospatial_pointing_1e-6_60k/checkpoint-34000"
# model_name = "/DATA/disk0/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
processor = AutoProcessor.from_pretrained(model_name)

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"inference/inference_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Define all test cases
test_cases = [
    {
        "name": "planning_task_multi_frame",
        "images": [f"/localfolder/code/LLaMA-Factory/data/data/ShareRobot/planning/rt_frames_success/rtx_frames_success_42/62_robo_set#episode_15700/frame_{i}.png" for i in range(12)],
        "question": "To work toward <placing a cup into a drawer>, what are the next five steps after completing 1-<approach the cup>?",
        "output_type": "text"
    },
    {
        "name": "planning_task_formatted_multi_frame",
        "images": [f"/localfolder/code/LLaMA-Factory/data/data/ShareRobot/planning/rt_frames_success/rtx_frames_success_42/62_robo_set#episode_15700/frame_{i}.png" for i in range(12)],
        "question": "To work toward <placing a cup into a drawer>, what are the next four steps?, Answer in the format 1-<>, 2-<>, ...",
        "output_type": "text"
    },
    {
        "name": "detection_task",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/grounding.jpg",
        "question": "detect hat with red color, delivering coordinates in plain text format \"[x1, y1, x2, y2]\".",
        "output_type": "bbox"
    },
    {
        "name": "affordance_task",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/affordance.jpg",
        "question": "You are a robot using the joint control. The task is <take the cup>. Please predict a possible affordance area of the end effector, delivering coordinates in plain text format \"[x1, y1, x2, y2]\".",
        "output_type": "bbox"
    },
    {
        "name": "pointing_task_green_mug",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/pointing.jpg",
        "question": "Point to empty space in the left of the green mug and format output as JSON with x, y coordinates",
        "output_type": "point"
    },
    {
        "name": "pointing_task_red_hat",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/grounding.jpg",
        "question": "Point to red hat and format output as JSON with x, y coordinates",
        "output_type": "point"
    },
    {
        "name": "spatial_reasoning_task",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/pointing.jpg",
        "question": "Identify 2 spots within the vacant space that's between the two mugs and format output as JSON with x, y coordinates",
        "output_type": "points"
    },
    {
        "name": "spatial_question_task",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/pointing.jpg",
        "question": "Can the green mug be put behind the blue mug? Answer Yes or No",
        "output_type": "text"
    }
]

def draw_bbox_on_image(image, coord, output_path):
    """Draw bounding box on image"""
    img_with_bbox = image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    draw.rectangle([(coord[0], coord[1]), (coord[2], coord[3])], outline="red", width=3)
    img_with_bbox.save(output_path)
    print(f"Saved image with bounding box to {output_path}")

def draw_trajectory_on_image(image, coord, output_path):
    """Draw trajectory points on image"""
    img_with_points = image.copy()
    draw = ImageDraw.Draw(img_with_points)
    
    # Draw each trajectory point
    for point in coord:
        x, y = point
        draw.ellipse([(x-3, y-3), (x+3, y+3)], fill="red", outline="red")
    
    # Connect points with lines to show trajectory
    if len(coord) > 1:
        draw.line([tuple(point) for point in coord], fill="yellow", width=2)
    
    img_with_points.save(output_path)
    print(f"Saved image with trajectory points to {output_path}")

def draw_points_on_image(image, coord, output_path):
    """Draw points on image"""
    img_with_points = image.copy()
    draw = ImageDraw.Draw(img_with_points)
    
    # Handle different coordinate formats
    if isinstance(coord, list) and len(coord) > 0:
        if isinstance(coord[0], dict):
            points = [list(v.values()) for v in coord]
        elif isinstance(coord[0], list):
            points = coord
        else:
            points = [coord]
    elif isinstance(coord, dict):
        points = [list(coord.values())]
    else:
        points = [coord]
    
    # Draw each point
    for i, point in enumerate(points):
        x, y = point
        draw.ellipse([(x-5, y-5), (x+5, y+5)], fill="red", outline="red")
        # Add number labels
        draw.text((x+8, y-8), str(i+1), fill="blue")
    
    img_with_points.save(output_path)
    print(f"Saved image with points to {output_path}")

# Run inference for all test cases
for i, test_case in enumerate(test_cases):
    print(f"\n{'='*50}")
    print(f"Running test case {i+1}/{len(test_cases)}: {test_case['name']}")
    print(f"{'='*50}")
    
    # Prepare message content
    content = []
    
    # Handle multiple images or single image
    if "images" in test_case:
        # Multiple images case (for planning tasks)
        for img_path in test_case["images"]:
            content.append({
                "type": "image",
                "image": img_path
            })
    else:
        # Single image case
        content.append({
            "type": "image",
            "image": test_case["image"]
        })
    
    # Add the question text
    content.append({
        "type": "text", 
        "text": test_case["question"]
    })
    
    # Prepare message
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=16000, temperature=0.5, do_sample=True, top_k=None)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Save text output
    if test_case["output_type"] == "text":
        output_text_path = os.path.join(output_dir, f"{test_case['name']}_output.txt")
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(f"Question: {test_case['question']}\n\n")
            f.write(f"Answer: {output_text[0]}\n")
    
    print(f"Question: {test_case['question']}")
    print(f"Answer: {output_text[0]}")
    
    # Process visual outputs based on type
    if test_case["output_type"] in ["bbox", "trajectory", "point", "points"] and image_inputs and len(image_inputs) > 0:
        try:
            # Try to parse the coordinates
            coord = json.loads(output_text[0])
            
            # Get the first image for visualization
            img = image_inputs[0]
            
            if test_case["output_type"] == "bbox" and len(coord) == 4:
                # Bounding box format: [x1, y1, x2, y2]
                image_output_path = os.path.join(output_dir, f"{test_case['name']}_bbox.jpg")
                draw_bbox_on_image(img, coord, image_output_path)
                
            elif test_case["output_type"] == "trajectory" and isinstance(coord, list):
                # Trajectory format: [[x1, y1], [x2, y2], ...]
                if isinstance(coord[0], dict):
                    coord = [list(v.values()) for v in coord]
                image_output_path = os.path.join(output_dir, f"{test_case['name']}_trajectory.jpg")
                draw_trajectory_on_image(img, coord, image_output_path)
                
            elif test_case["output_type"] in ["point", "points"]:
                # Point(s) format: {"x": x, "y": y} or [{"x": x, "y": y}, ...]
                image_output_path = os.path.join(output_dir, f"{test_case['name']}_points.jpg")
                draw_points_on_image(img, coord, image_output_path)
                
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            print(f"Could not parse coordinates for visualization: {e}")
            print(f"Raw output: {output_text[0]}")

print(f"\n{'='*50}")
print(f"All inference completed! Results saved in: {output_dir}")
print(f"{'='*50}")