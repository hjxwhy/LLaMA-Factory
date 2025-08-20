import os
import json
from PIL import Image
import shutil


json_path = "/localfolder/code/LLaMA-Factory/data/data/FSD-Dataset/Level-4-5-Dataset.json"
image_root = "/localfolder/data/FSD-Dataset/data"
image_save = "/localfolder/code/LLaMA-Factory/data/data/FSD-Dataset"

# with open(json_path, "r") as f:
#     fsd_datas = json.load(f)

# for data in fsd_datas:
#     image_path = data["image"]
#     image_path = image_path.removeprefix("bridge_data_v2/")

#     image_path = os.path.join(image_root, "raw", image_path)
#     if "raw/rtx" in image_path:
#         image_path = image_path.replace("raw/rtx", "openx_without_video")
#     if "droid" in image_path:
#         image_path = image_path.replace("raw/droid", "droid_wo_video")
#     if os.path.exists(image_path):
#         pass
#     else:
#         print(f"image_path not exists: {image_path}")
#         breakpoint()
    


#     data["images"] = [image_path.replace(image_root+"/", "")]
#     save_path = os.path.join(image_save, data["images"][0])

    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # shutil.copy(image_path, save_path)

# with open(os.path.join(image_save, "FSD-Dataset.json"), "w") as f:
#     json.dump(fsd_datas, f, indent=4)

# <Description>
# </Description>

# <Reasoning>
# </Reasoning>

# <Answer>
# </Answer>

# <|quad_start|>, <|quad_end|>

# {\n\"silver round lid\": [87, 150, 201, 230]\n} # grounding


with open(os.path.join(image_save, "FSD-Dataset.json"), "r") as f:
    datas = json.load(f)

valid_datas = []
for data in datas:
    conversations = data["conversations"]

    gpt = conversations[1]
    if "<Description>" in gpt["value"]:
        addition_prompt = "Description the scene within <Description>\n</Description> XML tag. Reasoning the process within <Reasoning>\n</Reasoning> XML tag. Answer the question within <Answer>\n</Answer> XML tag."
        conversations[0]["value"] =  conversations[0]["value"] + "\n" + addition_prompt
    
    for conversation in conversations:
        if "<|quad_start|>" in conversation["value"]:
            conversation["value"] = conversation["value"].replace("<|quad_start|>", "<point>").replace("<|quad_end|>", "</point>")

    if data["question_type"] == "grounding":
        try:
            ans = json.loads(gpt["value"])
            gpt["value"] = json.dumps(ans)
        except:
            print(f"gpt value is not a valid json: {gpt['value']}")
            continue

        conversation = conversations[0]
        conversation["value"] = conversation["value"] + "Please provide the bounding box coordinates in JSON format with the following structure:\n{\"object_name\": [x_min, y_min, x_max, y_max]}\nWhere the coordinates are in pixel values."
        
    valid_datas.append(data)

with open(os.path.join(image_save, "FSD-Dataset-fix.json"), "w") as f:
    json.dump(valid_datas, f, indent=4)
