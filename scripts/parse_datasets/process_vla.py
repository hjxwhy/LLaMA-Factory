import os
import json
import random

data_root = "/jfs/jensen/code/LLaMA-Factory/data/data/open-x/train"

# # # aloha_bi_play, aloha_mobile, droid, rdt, stanford_hydra_dataset_converted_externally_to_rlds, utaustin_mutex, viola
# data_names = ["aloha_bi_play", "aloha_mobile", "droid", "stanford_hydra_dataset_converted_externally_to_rlds", "utaustin_mutex", "viola"]
# dataset = []
# for data_name in data_names:
#     with open(os.path.join(data_root, f"train_datas_{data_name}.json"), "r") as f:
#         data = json.load(f)


#     for item in data:
#         item["images"] = [os.path.join("open-x", f.replace("images/", "videos/").lstrip("/")) for f in item["images"]]
#         item["messages"] = [
#             {
#                 "content": "".join(["<image>"] * len(item["images"])) + "\n" + message["content"] if message["role"] == "user" else message["content"],
#                 "role": message["role"],
#             }
#             for message in item["messages"]
#         ]
        
#     random.shuffle(data)
#     if data_name == "droid":
#         data = data[:len(data)//3]
#     dataset.extend(data)
        
# # aloha, 
# data_names = ["rdt", "aloha"]
# for data_name in data_names:
#     with open(os.path.join(data_root, f"train_datas_{data_name}.json"), "r") as f:
#         data = json.load(f)

#     for item in data:
#         item["images"] = [os.path.join("open-x", f.replace("images/", "videos/").lstrip("/")) for f in item["images"]]
#         fix_images = []
#         for image in item["images"]:
#             image_path = image.split("/")
#             fix_images.append(os.path.join(*image_path[:4],image_path[-3], image_path[-4], *image_path[-2:]))
#         item["images"] = fix_images
#         item["messages"] = [
#             {
#                 "content": "".join(["<image>"] * len(item["images"])) + "\n" + message["content"] if message["role"] == "user" else message["content"],
#                 "role": message["role"],
#             }
#             for message in item["messages"]
#         ]
#         # breakpoint()
#         dataset.append(item)



# data_names = ["berkeley_autolab_ur5", "bridge", "fractal20220817_data", "jaco_play"]
# for data_name in data_names:
#     with open(os.path.join(data_root, f"train_datas_{data_name}.json"), "r") as f:
#         data = json.load(f)
#     for item in data:
#         item["images"] = [os.path.join("open-x", f.lstrip("/")) for f in item["images"]]
#         item["messages"] = [
#             {
#                 "content": "".join(["<image>"] * len(item["images"])) + "\n" + message["content"] if message["role"] == "user" else message["content"],
#                 "role": message["role"],
#             }
#             for message in item["messages"]
#         ]
#     random.shuffle(data)
#     if data_name == "fractal20220817_data":
#         data = data[:len(data)//2]
#     dataset.extend(data)

# random.shuffle(dataset)
# print(len(dataset))
# with open(os.path.join(data_root, "train_datas_all.json"), "w") as f:
#     json.dump(dataset, f, indent=4)

with open(os.path.join(data_root, "train_datas_all.json"), "r") as f:
    data = json.load(f)

print(len(data))
from datasets import Dataset

dataset = Dataset.from_list(data)

dataset.to_json(os.path.join(data_root, "train_datas_all_hf.json"))



exit()

# with open("/jfs/jensen/code/LLaMA-Factory/data/data/open-x/train/Level-4-5-Dataset.json", "r") as f:
#     data = json.load(f)
# print(len(data))
# valid_data = []
# for item in data:
    
#     if "bridge_data_v2" in item["image"]:
#         continue
#     valid_data.append(item)
# print(len(valid_data))
# with open("/jfs/jensen/code/LLaMA-Factory/data/data/open-x/train/Level-4-5-Dataset-valid.json", "w") as f:
#     json.dump(valid_data, f, indent=4)
    

from datasets import Dataset, load_dataset, concatenate_datasets
# with open("/jfs/jensen/code/LLaMA-Factory/data/data/open-x/train/train_datas_all.json", "r") as f:
#     data = json.load(f)

# dataset = Dataset.from_list(data)

# vlm_dataset = load_dataset("/jfs/jensen/code/LLaMA-Factory/data/data/vlm_mix_robot_training_fix")["train"]
# print(len(vlm_dataset))
# vlm_dataset = vlm_dataset.filter(lambda x: (x["images"] is not None and len(x["images"]) > 0 and "spatial_qa" not in x["images"][0]) or (x["images"] is None or len(x["images"]) == 0))
# print(len(vlm_dataset))

# merge_dataset = concatenate_datasets([dataset, vlm_dataset]).shuffle()

def save_llava_dataset(dataset, output_path):
    """Save dataset in LLaVA format with multiple shards"""
    num_shards = 6
    os.makedirs(output_path, exist_ok=True)
    for i in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=i)
        shard.to_parquet(f"{output_path}/train-{i:05d}-of-{num_shards:05d}.parquet")

def filter_dataset_features(dataset, dataset_name):
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
# save_llava_dataset(merge_dataset, "/jfs/jensen/code/LLaMA-Factory/data/data/vlm_mix_robot_openx_training_v2")

vlm_dataset = load_dataset("/jfs/jensen/code/LLaMA-Factory/data/data/vlm_mix_robot_openx_training_v2")["train"]

with open("/jfs/jensen/code/LLaMA-Factory/data/data/FSD-Dataset/FSD-Dataset-fix.json", "r") as f:
    data = json.load(f)

def process_fsd_data(data):
    messages = []
    for item in data["conversations"]:
        messages.append({
            "content": item["value"],
            "role": "user" if item["from"] == "human" else "assistant",
        })
    return {
        "id": str(data["id"]) + data["question_type"],
        "messages": messages,
        "images_new": [os.path.join("FSD-Dataset", f) for f in data["images"]]
    }

fsd_ds = Dataset.from_list(data)
fsd_ds = fsd_ds.map(process_fsd_data, num_proc=16)
fsd_ds = fsd_ds.remove_columns(["images"])
fsd_ds = fsd_ds.rename_column("images_new", "images")
fsd_ds = filter_dataset_features(fsd_ds, "fsd_ds")

merge_dataset = concatenate_datasets([vlm_dataset, fsd_ds]).shuffle()
save_llava_dataset(merge_dataset, "/jfs/jensen/code/LLaMA-Factory/data/data/vlm_mix_robot_openx_training_v3")