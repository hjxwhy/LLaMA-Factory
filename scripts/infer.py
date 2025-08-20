import json
import os
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw
import numpy as np

# Model setup
model_name = "/DATA/disk0/saves/qwen2_5vl-7b/full/sft_mix_llava_onevision_robospatial_pointing_morewb_1e-6_100k/checkpoint-52000"
# model_name = "/DATA/disk0/Qwen2.5-VL-7B-Instruct"
# model_name = "/DATA/disk0/RoboBrain2.0-7B"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
processor = AutoProcessor.from_pretrained(model_name)

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"inference/inference_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Define all test cases
test_cases = [
    {
        "name": "pick_place",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/pick_place.jpg",
        "question": "Find the the black bowl on the wooden cabinet and the bounding box format is list of [x1, y1, x2, y2]",
        # "question": "The task is <pick up the black bowl on the wooden cabinet>. Please predict a possible affordance area of the end effector, delivering coordinates in plain text format \"[x1, y1, x2, y2]\".",
        "output_type": "bbox"
        # pick up the black bowl on the wooden cabinet and place it on the plate, 
    },
    {
        "name": "action_task",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/0020.png",
        "question": "Task: open middle drawer, FPS: 3, <state><pos><ACTION_32><ACTION_111><ACTION_101><pos><rot6d><ACTION_78><ACTION_44><ACTION_43><ACTION_11><ACTION_177><ACTION_144><rot6d><gripper><ACTION_255><gripper><state>\n<control_mode> end effector <control_mode>",
        "output_type": "text"
        # "Action: <ACTION_143><ACTION_203><ACTION_44><ACTION_124><ACTION_137><ACTION_172><ACTION_231><ACTION_165><ACTION_185><ACTION_66><ACTION_116><ACTION_118><ACTION_163><ACTION_203><ACTION_172><ACTION_162><ACTION_104><ACTION_111><ACTION_87><ACTION_150><ACTION_173>"
    },
    {
        "name": "spatial_qa_task_1",
        "image": "/localfolder/code/LLaMA-Factory/data/data/ShareRobot/spatial_qa/images/example_000012_image_00.png",
        "question": "There are four points marked with letters, which one shows the spring of the guitar capo Choices: \nA. A. \nB. B. \nC. C. \nD. D. \nPlease answer directly with only the letter of the correct option and nothing else.",
        "output_type": "text"
    },
    {
        "name": "traj_desc",
        "image": "/localfolder/code/LLaMA-Factory/data/data/ShareRobot/planning/track_human_label/approved/rtx_frames_success_11/49_bridge#episode_22254_smooth_curves.png",
        "question": "The image shows the motion trajectory of the robotic arm's end-effector (gripper) in 3D space, with green and blue representing the path from the starting to the ending positions. This trajectory corresponds to the future positions of the end-effector in 3D space, projected onto the current (initial) frame of the image. The red mark on the gripper indicates the starting point of the trajectory. Assuming the robotic arm's end-effector moves along the given trajectory, please describe the robotic task that can be accomplished by this motion.",
        "output_type": "text"
    },
    {
        "name": "traj_choice",
        "image": "/localfolder/code/LLaMA-Factory/data/data/ShareRobot/planning/track_human_label/approved/rtx_frames_success_11/49_bridge#episode_22254_smooth_curves.png",
        "question": "The image shows the motion trajectory of the robotic arm's end-effector (gripper) in 3D space, with green and blue representing the path from the starting to the ending positions. This trajectory corresponds to the future positions of the end-effector in 3D space, projected onto the current (initial) frame of the image. The red mark on the gripper indicates the starting point of the trajectory. Assuming the robotic arm's end-effector moves along the given trajectory, what will happen? Choices: A. Place orange cloth beside can of tomato sauce. B. Move orange cloth to front of tomato. C. Move orange cloth to front of can of tomato sauce.. D. Pick up orange cloth without placing it. Please answer directly with only the letter of the correct option and nothing else.",
        "output_type": "text"
    },
    {
        "name": "text_point",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/0000.png",
        "question": "图中有几种杯子款式?",
        "output_type": "text"
    },
    {
        "name": "point_task_1",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/0000.png",
        "question": "Point to four white cups in the image and format output as list of JSON with x, y coordinates",
        "output_type": "points"
    },
    {
        "name": "point_task_2",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/0000.png",
        "question": "Point to cup in the right of the green cup and format output as JSON with x, y coordinates",
        "output_type": "point"
    },
    {
        "name": "pitch_movement_1",
        "images": ["/localfolder/code/LLaMA-Factory/data/data/EmbodiedScan/scannet/posed_images/scene0505_00/02450.jpg", "/localfolder/code/LLaMA-Factory/data/data/EmbodiedScan/scannet/posed_images/scene0505_00/02350.jpg"],
        "question": "Identify the differences in camera pose when comparing these images. The movement should be relative to the first image. Note that the objects in the images are assumed to be static.\nPitch is the up-down rotation relative to the ground. Could you identify if the camera's pitch is rotating up or rotating down?",
        "output_type": "text"
    },
    {
        "name": "pitch_movement_2",
        "images": ["/localfolder/code/LLaMA-Factory/data/data/EmbodiedScan/scannet/posed_images/scene0505_00/02350.jpg", "/localfolder/code/LLaMA-Factory/data/data/EmbodiedScan/scannet/posed_images/scene0505_00/02450.jpg"],
        "question": "Identify the differences in camera pose when comparing these images. The movement should be relative to the first image. Note that the objects in the images are assumed to be static.\nPitch is the up-down rotation relative to the ground. Could you identify if the camera's pitch is rotating up or rotating down?",
        "output_type": "text"
    },
    {
        "name": "pitch_movement_3",
        "images": ["/localfolder/code/LLaMA-Factory/data/data/EmbodiedScan/mllm_spatial/visual_correspondence_dot_2_multichoice/v1_0/images/scene0629_02/140986_point222463_00050_01600_img1.jpg", "/localfolder/code/LLaMA-Factory/data/data/EmbodiedScan/mllm_spatial/visual_correspondence_dot_2_multichoice/v1_0/images/scene0629_02/140986_point222463_00050_01600_img2.jpg"],
        "question": "Identify the points that match in the images.\nIdentify which of the labeled points in second image is the same point as the annotated circle in first image.\nAnswer the question in single letter.",
        "output_type": "text"
    },
    {
        "name": "affordance_point",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/affordance.jpg",
        "question": "Please predict a possible affordance area of the cup and format output as JSON with x, y coordinates.",
        "output_type": "points"
    },
    {
        "name": "llava_next_task",
        "image": "/localfolder/code/LLaMA-Factory/data/data/LLaVA-NeXT-Data/images/000001.png",
        "question": "The cat the above the sofa? Answer Yes or No.",
        "output_type": "text"
    },
    {
        "name": "llava_next_task_1",
        "image": "/localfolder/code/LLaMA-Factory/data/data/EmbodiedScan/arkitscenes/Training/40753679/40753679_frames/lowres_wide/40753679_6790.132.png",
        "question": "Find the door and format output as JSON with x, y coordinates.",
        "output_type": "points"
    },
    {
        "name": "pointing_task_multi_items",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/image.jpg",
        "question": "Find farest clothes and format output as JSON with x, y coordinates.",
        "output_type": "point"
    },
    {
        "name": "pointing_task_1",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/000000001000.jpeg",
        "question": "Point to the the person wearing hats and format output as JSON with x, y coordinates",
        "output_type": "point"
    },
    {
        "name": "bbox_task_1",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/000000001000.jpeg",
        "question": "Output the bounding boxes for the leftmost person wearing red shirts, the bounding box format is list of [x1, y1, x2, y2]",
        "output_type": "bbox"
    },
    {
        "name": "bbox_task_2",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/000000001000.jpeg",
        "question": "Output the bounding boxes for the farthest person, the bounding box format is list of [x1, y1, x2, y2]",
        "output_type": "bbox"
    },
    {
        "name": "bbox_task_3",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/000000001000.jpeg",
        "question": "Output the bounding boxes for the closest person, the bounding box format is list of [x1, y1, x2, y2]",
        "output_type": "bbox"
    },
    {
        "name": "spatial_qa_task_1",
        "image": "/localfolder/code/LLaMA-Factory/data/data/ShareRobot/spatial_qa/images/example_000012_image_00.png",
        "question": "There are four points marked with letters, which one shows the spring of the guitar capo Choices: A. A. B. B. C. C. D. D. Please answer directly with only the letter of the correct option and nothing else.",
        "output_type": "text"
    },
    {
        "name": "spatial_qa_task_2",
        "image": "/localfolder/code/LLaMA-Factory/data/data/ShareRobot/spatial_qa/images/example_000000_image_00.png",
        "question": "If the yellow robot gripper follows the yellow trajectory, what will happen? Choices: A. Robot puts the soda on the wooden steps. B. Robot moves the soda in front of the wooden steps. C. Robot moves the soda to the very top of the wooden steps. D. Robot picks up the soda can and moves it up. Please answer directly with only the letter of the correct option and nothing else.",
        "output_type": "text"
    },
    {
        "name": "spatial_qa_task_3",
        "image": "/localfolder/code/LLaMA-Factory/data/data/ShareRobot/spatial_qa/images/example_000137_image_00.png",
        "question": "Assuming the top left corner is 0,0 and bottom right corner is 1000, 1000. What is at the point [y=718 x=273]? Choices: A. a helmet. B. a light blue Arc'teryx jacket. C. a puff jacket. D. snacks. Please answer directly with only the letter of the correct option and nothing else.",
        "output_type": "text",
        "answer": "B"
    },
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
        "question": "You are a robot using the joint control. The task is <hold the cup>. Please predict a possible affordance area of the end effector, delivering coordinates in plain text format \"[x1, y1, x2, y2]\".",
        "output_type": "bbox"
    },
    {
        "name": "pointing_task_green_mug",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/pointing.jpg",
        "question": "Point to empty space in the left of the green mug and format output as JSON with x, y coordinates",
        "output_type": "point"
    },
    {
        "name": "pointing_task_behind_green_mug",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/pointing.jpg",
        "question": "Point to empty space behind the green mug camera's perspective and format output as JSON with x, y coordinates",
        "output_type": "point"
    },
    {
        "name": "pointing_task_behind_blue_mug",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/pointing.jpg",
        "question": "Point to empty space behind the blue mug camera's perspective and format output as JSON with x, y coordinates",
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
        "question": "Can the green mug be put behind the blue mug camera's perspective? Answer Yes or No",
        "output_type": "text"
    },
    {
        "name": "spatial_question_task_2",
        "image": "/localfolder/code/LLaMA-Factory/tests/images/pointing.jpg",
        "question": "Can the green mug be put between the two mugs from the camera's perspective? Answer Yes or No",
        "output_type": "text"
    }
]

def draw_bbox_on_image(image, coords, output_path):
    """Draw bounding box(es) on image"""
    img_with_bbox = image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    
    # Handle both single bbox and multiple bboxes
    if isinstance(coords[0], list):
        # Multiple bounding boxes: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        bboxes = coords
    else:
        # Single bounding box: [x1, y1, x2, y2]
        bboxes = [coords]
    
    # Draw each bounding box
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
    for i, bbox in enumerate(bboxes):
        if len(bbox) == 4:
            color = colors[i % len(colors)]
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=color, width=3)
            # Add label with box number
            draw.text((bbox[0], bbox[1]-15), f"Box {i+1}", fill=color)
    
    img_with_bbox.save(output_path)
    print(f"Saved image with {len(bboxes)} bounding box(es) to {output_path}")

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
for i, test_case in enumerate(test_cases[:1]):
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
    generated_ids = model.generate(**inputs, max_new_tokens=16000, temperature=0.5, do_sample=False, top_k=None)
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
            
            if test_case["output_type"] == "bbox":
                # Bounding box format: [x1, y1, x2, y2] or [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
                if isinstance(coord, list) and len(coord) > 0:
                    # Check if it's a valid bbox format
                    if isinstance(coord[0], list) or (len(coord) == 4 and all(isinstance(x, (int, float)) for x in coord)):
                        image_output_path = os.path.join(output_dir, f"{test_case['name']}_bbox.jpg")
                        draw_bbox_on_image(img, coord, image_output_path)
                    else:
                        print(f"Invalid bounding box format: {coord}")
                
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
torch.cuda.empty_cache()
print(f"\n{'='*50}")
print(f"All inference completed! Results saved in: {output_dir}")
print(f"{'='*50}")