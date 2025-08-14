import json
import os
import re
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset, concatenate_datasets
from PIL import Image
corr_dot_multichoice_path = "/jfs/jensen/code/LLaMA-Factory/data/data/EmbodiedScan/mllm_spatial/visual_correspondence_dot_2_multichoice/v1_0/train_visual_correspondence_dot_2_multichoice.jsonl"
corr_dot_multichoice_datas = []
with open(corr_dot_multichoice_path, "r") as f:
    for line in f:
        data = json.loads(line)
        corr_dot_multichoice_datas.append(data)

print(len(corr_dot_multichoice_datas))

corr_dot_multichoice_dataset = Dataset.from_list(corr_dot_multichoice_datas)
# corr_dot_multichoice_dataset = corr_dot_multichoice_dataset.rename_column("image", "images")

def process_dot_multichoice_data(data):
    # process image paths
    image_paths = data["image"]
    image_paths = [os.path.join("EmbodiedScan/mllm_spatial/visual_correspondence_dot_2_multichoice/v1_0/images", f) for f in image_paths]

    # process conversations to messages
    messages = []
    map_role = {
        "human": "user",
        "gpt": "assistant"
    }
    for conversation in data["conversations"]:
        # 50% drop the Image-1, Image-2 tag in content
        value = conversation["value"]
        if np.random.rand() < 0.5:
            # Remove the first occurrence of "Image-1" and "Image-2" from the left
            value, n = re.subn(r"Image-(1|2):\s*\n?", "", value, count=2)
            value = value.lstrip()
            # Replace any remaining "Image-1" or "Image-2" with "first image" or "second image"
            value = re.sub(r"Image-1", "first image", value)
            value = re.sub(r"Image-2", "second image", value)
            if np.random.rand() < 0.5:
                value = value.replace("<image>\n", "<image>")
        role_from = conversation["from"]
        # 如果角色是"gpt"回答，提取答案
        if role_from == "gpt":
            # 优先匹配文本最右边引号中的内容
            matches = re.findall(r'[`\'"]([ABCD])[`\'"]', value)
            if matches:
                answer = matches[-1]  # 取最右边的匹配
            else:
                # 如果匹配不到引号中的内容，就整句话匹配A,B,C,D
                matches = re.findall(r'[ABCD]', value)
                if matches:
                    answer = matches[-1]  # 取最右边的匹配
                else:
                    answer = value.strip()
            value = answer
        else:
            value = value.strip() + "\nAnswer the question in single letter."
        messages.append({
            "content": value,
            "role": map_role[conversation["from"]]
        })
    return {
        "id": str(data["id"]),
        "messages": messages,
        "images": image_paths
    }

def process_camera_movement_data(data):
    # process image paths
    image_paths = data["image"]
    image_paths = [os.path.join("EmbodiedScan/scannet/posed_images", f) for f in image_paths]

    # process conversations to messages
    messages = []
    map_role = {
        "human": "user",
        "gpt": "assistant"
    }
    for conversation in data["conversations"]:
        # 50% drop the Image-1, Image-2 tag in content
        value = conversation["value"]
        if np.random.rand() < 0.5:
            # Remove the first occurrence of "Image-1" and "Image-2" from the left
            value, n = re.subn(r"Image-(1|2):\s*\n?", "", value, count=2)
            value = value.lstrip()
            # Replace any remaining "Image-1" or "Image-2" with "first image" or "second image"
            value = re.sub(r"Image-1", "first image", value)
            value = re.sub(r"Image-2", "second image", value)
            if np.random.rand() < 0.5:
                value = value.replace("<image>\n", "<image>")
        value = value.replace("`", "'")
        messages.append({
            "content": value,
            "role": map_role[conversation["from"]]
        })
    return {
        "id_str": str(data["id"]),
        "messages": messages,
        "images": image_paths
    }

corr_dot_multichoice_dataset = corr_dot_multichoice_dataset.map(process_dot_multichoice_data, num_proc=16)


camera_movement_dir = "/jfs/jensen/code/LLaMA-Factory/data/data/EmbodiedScan/mllm_spatial/camera_movement/v1_0"
camera_movement_files = os.listdir(camera_movement_dir)
camera_movement_files = [os.path.join(camera_movement_dir, f) for f in camera_movement_files]

camera_movement_datas = []
for file in camera_movement_files:
    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)
            camera_movement_datas.append(data)

camera_movement_datas = camera_movement_datas[::5]
print(len(camera_movement_datas))

camera_movement_dataset = Dataset.from_list(camera_movement_datas)
camera_movement_dataset = camera_movement_dataset.map(process_camera_movement_data, num_proc=16)

def filter_dataset_features( dataset, dataset_name):
    """Filter dataset to keep only required features: [id, tools, images, audios, messages]"""
    if len(dataset) == 0:
        print(f"{dataset_name}: 0 samples")
        return dataset
        
    required_features = ["id", "tools", "images", "audios", "messages"]
    current_features = list(dataset.features.keys())
    
    # Find features to remove
    features_to_remove = [f for f in current_features if f not in required_features]
    
    if features_to_remove:
        print(f"Removing features {features_to_remove} from {dataset_name}")
        dataset = dataset.remove_columns(features_to_remove)
    
    print(f"{dataset_name}: {len(dataset)} samples")
    return dataset

camera_movement_dataset = camera_movement_dataset.remove_columns("id")
camera_movement_dataset = camera_movement_dataset.rename_column("id_str", "id")
corr_dot_multichoice_dataset = filter_dataset_features(corr_dot_multichoice_dataset, "corr_dot_multichoice_dataset")
camera_movement_dataset = filter_dataset_features(camera_movement_dataset, "camera_movement_dataset")


output_dir = "/jfs/jensen/code/LLaMA-Factory/data/data/EmbodiedScan/mllm_spatial"

# merged_dataset = concatenate_datasets([corr_dot_multichoice_dataset, camera_movement_dataset])
import gc
gc.collect()
# Convert id features to string

# merged_dataset.to_json(f"{output_dir}/mllm_spatial_v1_0.json", num_proc=16, force_ascii=False, indent=4)


from qwen_vl_utils import smart_resize

end_effector_path = "/jfs/jensen/code/LLaMA-Factory/data/data/ShareRobot/conversations_with_validation.json"
with open(end_effector_path, "r") as f:
    end_effector_datas = json.load(f)

end_effector = []
for i, data in enumerate(end_effector_datas):
    if "is_end_effector" not in data or data["is_end_effector"] == False:
        continue
    points = data["points"]
    x1, y1, x2, y2 = points
    image_path = data["images"][0].replace("/home/unitree/remote_jensen", "/jfs/jensen/code/LLaMA-Factory/data/data")
    image = Image.open(image_path)
    width, height = image.size
    height_resize, width_resize = smart_resize(height, width)
    x1_resize, y1_resize, x2_resize, y2_resize = x1 / width * width_resize, y1 / height * height_resize, x2 / width * width_resize, y2 / height * height_resize

    prompt = "Point robot arm two end-effector grippers. The answer should follow the json format: [{\"point\": <point>}, ...], points are in [x, y] format."
    points_dict = [
        {
            "point": [int(x1_resize), int(y1_resize)],
        },
        {
            "point": [int(x2_resize), int(y2_resize)],
        }
    ]
    messages = [
        {
            "content": "<image>" + prompt,
            "role": "user"
        },
        {
            "content": json.dumps(points_dict),
            "role": "assistant"
        }
    ]
    end_effector.append({
        "id": "end_effector_" + str(i),
        "messages": messages,
        "images": [image_path.replace("/jfs/jensen/code/LLaMA-Factory/data/data/", "")]
    })

end_effector_dataset = Dataset.from_list(end_effector)


task_split = "/jfs/jensen/code/LLaMA-Factory/data/data/ShareRobot/gemini_checkpoint_with_end_effort.json"
with open(task_split, "r") as f:
    task_split_datas = json.load(f)["results"]

task_split_datas = [data for data in task_split_datas if data["model_response"] is not None and "\"label\": \"start\"" not in data["model_response"]]

task_split_dataset = []
for data in task_split_datas:
    image_path = data["image_path"].replace("/home/jensen/remote_jensen/huangjianxin", "/jfs/jensen/code/LLaMA-Factory/data/data")
    image = Image.open(image_path)
    image.verify()
    width, height = image.size
    height_resize, width_resize = smart_resize(height, width)
    
    # prompt = data["prompt"]
    task = data["task"]

    prompt = f"""You need to help me point out the starting point and end point of the task on the image.
For example, for this task: "Put the spoon to the right of the cloth.", 
First you need to determine that the starting location of this task is label: spoon, and the end location is label: the right of the cloth.
Then point to the starting location and the end location on the image.
The answer should follow the json format: [{{"point": <point>, "label": <label1>}}, ...].
The point are in [x, y] format.
Now you task is
"{task}"
"""
    model_response = data["model_response"]
    
    # Extract JSON content from markdown code blocks
    if "```json" in model_response:
        start_idx = model_response.find("```json") + len("```json")
        end_idx = model_response.find("```", start_idx)
        if end_idx != -1:
            model_response = model_response[start_idx:end_idx].strip()
        else:
            model_response = model_response[start_idx:].strip()
    elif "```" in model_response:
        start_idx = model_response.find("```") + len("```")
        end_idx = model_response.find("```", start_idx)
        if end_idx != -1:
            model_response = model_response[start_idx:end_idx].strip()
        else:
            model_response = model_response[start_idx:].strip()
    model_response = model_response.replace("[[", "[")
    try:
        model_response = json.loads(model_response)
    except:
        print(f"Error parsing model response: {model_response}")
        continue

    points_dict = [
        {
            "point": [int(point["point"][1] / 1000 * width_resize), int(point["point"][0] / 1000 * height_resize)],
            "label": point["label"]
        }
        for point in model_response
    ]
    messages = [
        {
            "content": "<image>" + prompt,
            "role": "user"
        },
        {
            "content": json.dumps(points_dict),
            "role": "assistant"
        }
    ]
    task_split_dataset.append({
        "id": "task_split_" + str(i),
        "messages": messages,
        "images": [image_path.replace("/jfs/jensen/code/LLaMA-Factory/data/data/", "")]
    })

task_split_dataset = Dataset.from_list(task_split_dataset)


print(len(task_split_dataset))



frames_path = "/jfs/jensen/code/LLaMA-Factory/data/data/ShareRobot/planning/rt_frames_success"
track_human_label = "/jfs/jensen/code/LLaMA-Factory/data/data/ShareRobot/planning/track_human_label/approved"

frames = os.listdir(track_human_label)
track_infer_dataset = []
for frame in frames:
    rtx_desc_path = os.path.join(frames_path, frame, "rtx_desc.json")
    with open(rtx_desc_path, "r") as f:
        rtx_desc = json.load(f)
    
    images_files = os.listdir(os.path.join(track_human_label, frame))
    images_files = [os.path.join(track_human_label, frame, f) for f in images_files if f.endswith(".png")]
    images_files = sorted(images_files)

    for image_file in images_files:
        image = Image.open(image_file)
        image.verify()

        episode_id = image_file.split("/")[-1].split(".")[0].replace("_smooth_curves", "")
        
        task = rtx_desc[episode_id]

        prompt = f"""The image shows the motion trajectory of the robotic arm's end-effector (gripper) in 3D space, with green and blue representing the path from the starting to the ending positions. This trajectory corresponds to the future positions of the end-effector in 3D space, projected onto the current (initial) frame of the image. The red mark on the gripper indicates the starting point of the trajectory. Assuming the robotic arm's end-effector moves along the given trajectory, please describe the robotic task that can be accomplished by this motion."""

        messages = [
            {
                "content": "<image>" + prompt,
                "role": "user"
            },
            {
                "content": f"The task is: {task}",
                "role": "assistant"
            }
        ]
        track_infer_dataset.append({
            "id": "track_infer",
            "messages": messages,
            "images": [image_file.replace("/jfs/jensen/code/LLaMA-Factory/data/data/", "")]
        })

track_infer_dataset = Dataset.from_list(track_infer_dataset)
print(len(track_infer_dataset))
## with open("/jfs/jensen/code/LLaMA-Factory/data/data/ShareRobot/planning/track_human_label/track_infer_dataset.json", "w") as f:
##     json.dump(track_infer_dataset, f, indent=4)
        

import random

fake_option_path = "/jfs/jensen/code/LLaMA-Factory/data/data/ShareRobot/planning/track_human_label/track_infer_dataset_fake_task.json"
with open(fake_option_path, "r") as f:
    fake_option_datas = json.load(f)

prompt = f"""The image shows the motion trajectory of the robotic arm's end-effector (gripper) in 3D space, with green and blue representing the path from the starting to the ending positions. This trajectory corresponds to the future positions of the end-effector in 3D space, projected onto the current (initial) frame of the image. The red mark on the gripper indicates the starting point of the trajectory. Assuming the robotic arm's end-effector moves along the given trajectory, what will happen?"""

fake_option_qa = []
for data in fake_option_datas:
    fake_tasks = json.loads(data["fake_task"])
    
    fake_task_list = []
    if "data" in fake_tasks:
        fake_tasks = fake_tasks["data"]
    for fake_task in fake_tasks:
        fake_task_list.extend(list(fake_task.values()))

    
    true_task = data["messages"][-1]["content"]
    true_task = true_task.split("The task is: ")[1] + "."
    print(true_task)

    task_tuple = list(zip([True]+[False]*len(fake_task_list), [true_task]+fake_task_list))
    random.shuffle(task_tuple)
    
    option = ["A", "B", "C", "D"]
    option_list = []
    for op, (is_true, task) in zip(option, task_tuple):
        if is_true:
            ans = op
        tmp = op+". "+task
        if tmp[-1] != ".":
            tmp = tmp + "."
        option_list.append(tmp+" ")

    option_list = "".join(option_list)

    messages = [
        {
            "content": "<image>" + prompt+" Choices: " + option_list + "Please answer directly with only the letter of the correct option and nothing else.",
            "role": "user"
        },
        {
            "content": ans,
            "role": "assistant"
        }
    ]
    fake_option_qa.append({
        "id": data["id"],
        "messages": messages,
        "images": data["images"]
    })


fake_option_dataset = Dataset.from_list(fake_option_qa)

# with open("/jfs/jensen/code/LLaMA-Factory/data/data/ShareRobot/planning/track_human_label/track_infer_dataset_fake_option_qa.json", "w") as f:
#     json.dump(fake_option_qa, f, indent=4)

traj_llamafactory_data = []
traj_points = "/jfs/jensen/code/LLaMA-Factory/data/data/ShareRobot/gemini_traj_all_results.json"
valid_sample = "/jfs/jensen/code/LLaMA-Factory/data/data/ShareRobot/progress.json"
with open(valid_sample, "r") as f:
    valid_sample_datas = json.load(f)

valid_sample_datas = set(valid_sample_datas["approved"])


with open(traj_points, "r") as f:
    traj_points_datas = json.load(f)
# breakpoint()
traj_points_datas = [data for data in traj_points_datas if ("visualization_path" in data) and  data["visualization_path"] is not None and (os.path.join("/home/jensen/code/cookbook/examples", data["visualization_path"])) in valid_sample_datas]

# 第一种情况没有两个轨迹，只有机械臂到目标点
# 第二种情况有两个轨迹，机械臂到抓取物体，然后到目标点
# print(traj_points_datas[0])

# Your are a robot arm, your task is {task}.
# First point to the robot end-effector, then point to the manipulation object and the target location, format in [x, y].
# Second a trajectory from the robot end-effector to the manipulation object.
# Third a trajectory from the manipulation object to the target location.
# If the manipulation object and target location are the same, only return a trajectories from the robot end-effector to the target location.
# The trajectory should follow the json format: [{\"point\": <point>}, ...].\nThe points are in [x, y] format.

from qwen_vl_utils import smart_resize
for data in traj_points_datas:
    # prompt = data["prompt"]
    # prompt = prompt.split("\n")
    # # 第一种情况没有两个轨迹，只有机械臂到目标点
    # # 第二种情况有两个轨迹，机械臂到抓取物体，然后到目标点
    # # if " in " not in prompt[1:4][2]:
    # print(prompt[1:4])
    #     # break
    image_path = data["image_path"].replace("/home/jensen/remote_jensen/huangjianxin", "/jfs/jensen/code/LLaMA-Factory/data/data")
    image = Image.open(image_path)
    width, height = image.size
    height_resize, width_resize = smart_resize(height, width)
    # image = image.resize((width_resize, height_resize))

    prompt = data["prompt"]
    # Extract all coordinate data from prompt and track their positions
    import re
    
    # Find all coordinate patterns [y, x] in the prompt
    coordinate_pattern = r'\[(\d+),\s*(\d+)\]'
    matches = re.finditer(coordinate_pattern, prompt)
    
    coordinates_info = []
    for match in matches:
        start_pos = match.start()
        end_pos = match.end()
        y_coord = int(match.group(1))
        x_coord = int(match.group(2))
        full_match = match.group(0)
        
        coordinates_info.append({
            'coordinates': [x_coord, y_coord],
            'start_position': start_pos,
            'end_position': end_pos,
            'original_text': full_match
        })
    
    for coord_info in coordinates_info:
        x_coord, y_coord = coord_info['coordinates']
        x_coord_resize, y_coord_resize = x_coord / 1000 * width_resize, y_coord / 1000 * height_resize
        coord_str = f"[{int(x_coord_resize)}, {int(y_coord_resize)}]"
        prompt = prompt.replace(coord_info['original_text'], coord_str)

    prompt = prompt.replace(" \"label\": <label1>},", "").replace("[y, x]", "[x, y]").replace(" normalized to 0-1000", "")
    # print(prompt)

    model_response = data["model_response"]
    model_response = model_response.replace("```json\n", "").replace("```", "")
    model_response = json.loads(model_response)
    for i, point in enumerate(model_response):
        point.pop("label")
        y,x = point["point"]
        y_resize, x_resize = int(y / 1000 * height_resize), int(x / 1000 * width_resize)
        model_response[i]["point"] = [x_resize, y_resize]
    # print(model_response)

    messages = [
        {
            "content": "<image>" + prompt,
            "role": "user"
        },
        {
            "content": json.dumps(model_response),
            "role": "assistant"
        }
    ]
    
    traj_llamafactory_data.append({
        "id": "gemini_traj",
        "messages": messages,
        "images": [image_path.replace("/jfs/jensen/code/LLaMA-Factory/data/data/", "")]
    })

    # break

traj_llamafactory_dataset = Dataset.from_list(traj_llamafactory_data)

merged_dataset = concatenate_datasets([corr_dot_multichoice_dataset, camera_movement_dataset, traj_llamafactory_dataset, task_split_dataset, end_effector_dataset, fake_option_dataset, track_infer_dataset])
merged_dataset.to_json(f"{output_dir}/mllm_spatial_v1_0.json", num_proc=16, force_ascii=False, indent=4)