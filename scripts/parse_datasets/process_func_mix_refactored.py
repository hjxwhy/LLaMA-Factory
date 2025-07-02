import json
import os
import sys
import argparse
import ast
import re
import logging
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset
import random
import gc
from copy import deepcopy

# Add the common directory to the path to import config
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from ..common.config import ConfigManager, DataMixConfig

try:
    from PIL import Image
    from qwen_vl_utils import smart_resize
    QWEN_VL_AVAILABLE = True
except ImportError:
    QWEN_VL_AVAILABLE = False
    logging.warning("qwen_vl_utils not available. Qwen VL mixing functionality will be limited.")

class DataMixer:
    """Class to handle data mixing operations with configurable paths"""
    
    def __init__(self, config: DataMixConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def set_seed(self, seed=0):
        """Set random seed for reproducibility"""
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ===== Original DataMixer methods =====
    def load_unitree_dataset(self):
        """Load and process Unitree dataset"""
        all_unitree_data = []
        
        # Load audio user data
        if not os.path.exists(self.config.unitree_audio_user_path):
            self.logger.warning(f"Unitree audio user path not found: {self.config.unitree_audio_user_path}")
            return Dataset.from_list([])
            
        uuid_files = os.listdir(self.config.unitree_audio_user_path)
        for file in uuid_files:
            conversation_file = os.path.join(
                self.config.unitree_audio_user_path, 
                file, 
                "conversation_v2_noised_music.json"
            )
            
            if not os.path.exists(conversation_file):
                continue
                
            with open(conversation_file, "r") as f:
                data = json.load(f)
                
            if len(data[0]["messages"]) < 2 or len(data[0]["messages"]) % 2 != 1:
                print(f"skip {file} because of length {len(data[0]['messages'])}")
                continue
                
            for i in range(len(data[0]["audios"])):
                data[0]["audios"][i] = os.path.join("unitree_train/common_prompts/audio_user", data[0]["audios"][i])
                
            messages = []
            for msg in data[0]["messages"]:
                messages.append({
                    "content": msg["content"],
                    "role": msg["role"]
                })
            
            data[0]["messages"] = messages
            data[0]["id"] = "unitree"
            all_unitree_data.extend(data)

        # Load G1 instruct data
        if os.path.exists(self.config.unitree_g1_instruct_path):
            with open(self.config.unitree_g1_instruct_path, "r") as f:
                data = json.load(f)

            for item in data:
                for i in range(len(item["audios"])):
                    item["audios"][i] = os.path.join("unitree_train/common_prompts", item["audios"][i])
                    
                messages = []
                for msg in item["messages"]:
                    messages.append({
                        "content": msg["content"],
                        "role": msg["role"]
                    })

                item["messages"] = messages
                item["id"] = "unitree"
                
            all_unitree_data.extend(data)
        
        return Dataset.from_list(all_unitree_data)

    def convert_conv2messages(self, example):
        """Convert conversation format to messages format"""
        messages = []
        role_map = {
            "human": "user",
            "gpt": "assistant",
            "system": "system"
        }
        for item in example["conversations"]:
            role = role_map[item["from"]] if item["from"] in role_map else item["from"]
            messages.append({"content": item["value"], "role": role})
            
        if "tools" in example:
            tools = json.loads(example["tools"])
            if tools is not None and len(tools) < 1:
                example["tools"] = None
                
        example["messages"] = messages
        example["id"] = "glaive_toolcall"
        return example

    def convert_xlam2messages(self, example):
        """Convert XLAM format to messages format"""
        messages = []
        messages.append({"content": example["query"], "role": "user"})
        messages.append({"content": example["answers"], "role": "function_call"})
        return {"messages": messages, "id": "xlam_toolcall_"+str(example["id"])}

    def save_llava_dataset(self, dataset, output_path):
        """Save dataset in LLaVA format with multiple shards"""
        num_shards = 6
        os.makedirs(output_path, exist_ok=True)
        for i in range(num_shards):
            shard = dataset.shard(num_shards=num_shards, index=i)
            shard.to_parquet(f"{output_path}/train-{i:05d}-of-{num_shards:05d}.parquet")

    def filter_dataset_features(self, dataset, dataset_name):
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

    def remove_audio_tag(self, example):
        """Remove audio tags from messages"""
        for message in example["messages"]:
            if message["role"] == "user" and "<audio>" in message["content"]:
                message["content"] = message["content"].replace("<audio>", "")
        return example

    def load_json_dataset(self, file_path):
        """Load JSON dataset and process messages for specific file formats"""
        if not os.path.exists(file_path):
            self.logger.warning(f"Dataset file {file_path} not found")
            return Dataset.from_list([])
            
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Process messages to convert content lists to strings for specific files
        if "uniree_toolcall_conversation_v2" in file_path:
            for item in data:
                if "messages" in item and item["messages"] is not None:
                    for message in item["messages"]:
                        if isinstance(message.get("content"), list):
                            # Convert content list to string by extracting text
                            content_str = ""
                            for content_item in message["content"]:
                                if isinstance(content_item, dict) and content_item.get("type") == "text":
                                    content_str += content_item.get("text", "")
                            message["content"] = content_str
            data = data * 3
            
        return Dataset.from_list(data)

    def get_tools(self, dataset):
        """Extract tools from dataset"""
        tools = []
        for example in dataset:
            if "tools" in example and example["tools"] is not None:
                tools.extend(json.loads(example["tools"]))
        return tools

    def add_random_tools_to_samples(self, dataset, all_tools, num_samples=10000):
        """Add random tools to selected samples"""
        if len(all_tools) == 0:
            self.logger.warning("No tools available for adding to samples")
            return dataset.select(range(min(num_samples, len(dataset))))
            
        num_samples = min(num_samples, len(dataset))
        
        total_len = len(dataset)
        indices = list(range(total_len))
        random.shuffle(indices)
        selected_indices = indices[:num_samples]
        
        sampled_dataset = dataset.select(selected_indices)
        
        tools_for_samples = []
        for _ in range(num_samples):
            num_tools = random.randint(1, min(5, len(all_tools)))
            selected_tools = random.sample(all_tools, num_tools)
            tools_for_samples.append(json.dumps(selected_tools))
        
        sampled_dataset = sampled_dataset.add_column("tools", tools_for_samples)
        return sampled_dataset

    def revert_role_content(self, example):
        """Revert role content order"""
        new_messages = []
        for item in example["messages"]:
            new_messages.append({"content": item["content"], "role": item["role"]})
        example["messages_new"] = new_messages
        return example

    def add_system_prompt(self, example, system_prompts_list):
        """Add random system prompt to example"""
        if len(system_prompts_list) > 0:
            random_system_prompt = random.choice(system_prompts_list)
            new_messages = [random_system_prompt] + example["messages"]
            example["messages"] = new_messages
        return example

    # ===== Random Combine functionality =====
    def load_fight_qa_dataset(self):
        """Load fight QA dataset"""
        if not os.path.exists(self.config.unitree_fight_qa_path):
            self.logger.warning(f"Fight QA path not found: {self.config.unitree_fight_qa_path}")
            return Dataset.from_list([])
            
        dataset = self.load_json_dataset(self.config.unitree_fight_qa_path)
        if len(dataset) == 0:
            return dataset

        data = dataset.to_list()
        for i in range(len(data)):
            data[i]["id"] = "fight_qa"
            data[i]["audios"] = [os.path.join("unitree_train", _) for _ in data[i]["audios"]]
        return Dataset.from_list(data)

    def load_voiceassistant_dataset(self):
        """Load voice assistant dataset"""
        if not os.path.exists(self.config.voice_assistant_path):
            self.logger.warning(f"Voice assistant path not found: {self.config.voice_assistant_path}")
            return Dataset.from_list([])
            
        with open(self.config.voice_assistant_path, "r") as f:
            # This file is jsonl, not json.
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
        return Dataset.from_list(data)

    def combine_conversations(self, original_conversation, voice_assistant_conversations):
        """Combine original conversation with voice assistant conversations"""
        result = original_conversation.copy()
        
        # Find valid insertion points where we can insert complete conversation pairs
        valid_insertion_points = []
        
        # Check if messages follow the pattern user->assistant
        for i in range(0, len(result["messages"]) - 1):
            if i+1 < len(result["messages"]):
                # Look for user->assistant pattern, we can insert after this pattern
                if (result["messages"][i]["role"] == "user" and 
                    result["messages"][i+1]["role"] == "assistant"):
                    valid_insertion_points.append(i-1)  # Insert after the assistant response
        
        # If we don't have valid insertion points, use fallback
        if not valid_insertion_points:
            # Try to find a position after system message if one exists
            for i, msg in enumerate(result["messages"]):
                if msg["role"] == "system" and i + 1 < len(result["messages"]):
                    valid_insertion_points = [i]
                    break
            
            # If still no valid points, we can't insert properly
            if not valid_insertion_points:
                return result
        
        # Determine how many VA conversations we can insert
        num_insertion_points = min(len(voice_assistant_conversations), len(valid_insertion_points))
        
        if num_insertion_points == 0:
            return result
        
        # Randomly select insertion points
        selected_points = random.sample(valid_insertion_points, num_insertion_points)
        
        # Track audio insertion info
        audio_insertions = []
        
        # Insert voice assistant conversations at selected points
        for i, insert_idx in enumerate(selected_points):
            if i >= len(voice_assistant_conversations):
                break
                
            va_conv = voice_assistant_conversations[i]
            
            # Ensure voice assistant conversation starts with user and follows user-assistant pattern
            va_messages = []
            for j in range(0, len(va_conv["messages"]), 2):
                if j + 1 < len(va_conv["messages"]):
                    if va_conv["messages"][j]["role"] == "user" and va_conv["messages"][j+1]["role"] == "assistant":
                        va_messages.append(va_conv["messages"][j])
                        va_messages.append(va_conv["messages"][j+1])
            
            if not va_messages:
                continue
            
            # Insert messages
            result["messages"][insert_idx+1:insert_idx+1] = va_messages
            
            # Calculate audio insertion position - count user messages before this point
            audio_pos = 0
            for j in range(insert_idx + 1):
                if result["messages"][j]["role"] == "user":
                    audio_pos += 1
            
            # Count how many user messages are in the VA conversation we're inserting
            va_user_msg_count = sum(1 for msg in va_messages if msg["role"] == "user")
            
            # Store audio insertion information
            audio_insertions.append({
                "position": audio_pos,
                "audios": va_conv["audios"][:va_user_msg_count]  # Only include audios for user messages we kept
            })
        
        # Sort audio insertions by position in ascending order
        audio_insertions.sort(key=lambda x: x["position"])
        
        # Insert audios at the tracked positions
        offset = 0
        for insertion in audio_insertions:
            pos = insertion["position"] + offset
            result["audios"][pos:pos] = insertion["audios"]
            offset += len(insertion["audios"])
        
        return result

    def save_dataset(self, combined_data, output_path):
        """Save the combined dataset to a JSON file"""
        with open(output_path, "w") as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        print(f"Combined dataset saved to {output_path}")

    # ===== Qwen VL Mix functionality =====
    def convert_sharerobot(self, example):
        """Convert ShareRobot format"""
        assert len(example["conversations"]) == 2
        assert isinstance(example["conversations"][1]["value"], str)
        example["messages"] = [
            {"content": example["conversations"][0]["value"].replace("<image> ", "<image>"), "role": "user"},
            {"content": example["conversations"][1]["value"], "role": "assistant"}
        ]
        if example["messages"][1]["content"] in ["no", "No", "yes", "Yes"]:
            example["messages"][0]["content"] = example["messages"][0]["content"] + "\nAnswer yes or no."
        
        # Fix step mismatch - check how many steps are actually in the answer
        user_content = example["messages"][0]["content"]
        assistant_content = example["messages"][1]["content"]
        
        # Count steps in assistant's answer by looking for the pattern "number-<...>"
        steps_in_answer = len(re.findall(r'\d+-<[^>]+>', assistant_content))
        
        # Check if user is asking for a specific number of steps
        five_step_patterns = [
            r'what are the next five things to do',
            r'what are the next five steps after completing',
            r'what are the next five tasks',
            r'what are the next five steps', 
            r'what\'s the next set of five steps',
            r'next set of five steps'
        ]
        
        # General patterns with capture group for the number
        number_step_patterns = [
            r'what are the next (\w+) things to do',
            r'what are the next (\w+) steps after completing',
            r'what are the next (\w+) tasks',
            r'what are the next (\w+) steps',
            r'what\'s the next set of (\w+) steps',
            r'next set of (\w+) steps'
        ]
        
        # Number words mapping
        number_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        number_words_reverse = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
        
        # Check for exact "five steps" patterns
        found_five_pattern = False
        for pattern in five_step_patterns:
            if re.search(pattern, user_content, re.IGNORECASE) and steps_in_answer != 5:
                found_five_pattern = True
                if 0 <= steps_in_answer <= 10:
                    # Replace "five" with actual step count
                    replacement = pattern.replace("five", number_words_reverse[steps_in_answer])
                    user_content = re.sub(pattern, replacement, user_content, flags=re.IGNORECASE)
                    example["messages"][0]["content"] = user_content
                break
        
        # If no "five steps" pattern was found, check for general number patterns
        if not found_five_pattern:
            for pattern in number_step_patterns:
                match = re.search(pattern, user_content, re.IGNORECASE)
                if match:
                    requested_steps_word = match.group(1).lower()
                    requested_steps = -1
                    
                    if requested_steps_word in number_words:
                        requested_steps = number_words[requested_steps_word]
                    elif requested_steps_word.isdigit():
                        requested_steps = int(requested_steps_word)
                    
                    if requested_steps != -1 and requested_steps != steps_in_answer and steps_in_answer > 0:
                        if 1 <= steps_in_answer <= 10:
                            # Create replacement by substituting the number word
                            original_text = match.group(0)
                            replacement = original_text.replace(requested_steps_word, number_words_reverse[steps_in_answer])
                            user_content = user_content.replace(original_text, replacement)
                            example["messages"][0]["content"] = user_content
                    break
        
        return example

    def process_robospatial(self, example):
        """Process RoboSpatial dataset"""
        if "matterport3d" in example["images"]:
            image_path = "/".join(example["images"].split("/")[-3:])
            example["images"] = [f"EmbodiedScan/matterport3d/{image_path}"]
        elif "arkitscenes" in example["images"]:
            example["images"] = [example["images"].replace("/home/jensen/remote_jensen/huangjianxin/", "")]
        else:
            example["images"] = [example["images"].replace("/home/jensen/remote_jensen/huangjianxin//", "")]
        
        if "<image>" not in example["messages"][0]["content"] and len(example["images"]) > 0:
            example["messages"][0]["content"] = "<image>" + example["messages"][0]["content"]
        return example

    def process_robospatial_v2(self, example):
        """Process RoboSpatial dataset v2"""
        example["images"] = [example["image_path"].replace("/home/jensen/remote_jensen/huangjianxin/", "")]
        split_path = example["images"][0].split("/")
        if "matterport" in example["images"][0] and split_path.count(split_path[-3]) > 1:
            example["images"][0] = os.path.join("EmbodiedScan/matterport3d", *example["images"][0].split("/")[-3:])
        example["messages"] = [{"content": example["question"], "role": "user"}, {"content": example["answer"], "role": "assistant"}]
        
        if example["qa_type"] == "unary_spatial_context" and example["images"] and QWEN_VL_AVAILABLE:
            remove_content = "The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points."
            
            example["messages"][0]["content"] = example["messages"][0]["content"].replace(remove_content, "")
            
            try:
                answer_content = example["messages"][-1]["content"]
                answer_content = ast.literal_eval(answer_content)
            except Exception as e:
                self.logger.warning(f"Error parsing answer content {answer_content}: {e}")
                return example
            
            images_path = example["images"]
            try:
                image = Image.open(os.path.join(self.config.llama_factory_root, "data/data", images_path[0]))
                w, h = image.width, image.height
            except Exception as e:
                self.logger.warning(f"Error loading image {images_path[0]}: {e}")
                return example

            resized_h, resized_w = smart_resize(h, w)
            abs_coords = []
            for point in answer_content:
                x = point[0] * resized_w
                y = point[1] * resized_h
                abs_coords.append((int(x), int(y)))

            example["messages"][-1]["content"] = json.dumps(abs_coords)
        return example

    def process_pointing(self, example):
        """Process pointing dataset"""
        if "<image>" not in example["messages"][0]["content"] and len(example["images"]) > 0:
            example["messages"][0]["content"] = "<image>" + example["messages"][0]["content"]
        return example

    # ===== Main processing methods =====
    def generate_unitree_mask_history(self):
        """Generate unitree_mask_history.json if it doesn't exist"""
        if os.path.exists(self.config.unitree_mask_history_path):
            self.logger.info(f"unitree_mask_history.json already exists at {self.config.unitree_mask_history_path}")
            return
            
        self.logger.info("Generating unitree_mask_history.json...")
        
        # Load required datasets
        unitree_toolcall_dataset = self.load_json_dataset(self.config.unitree_toolcall_path)
        if len(unitree_toolcall_dataset) == 0:
            self.logger.warning("Cannot generate unitree_mask_history.json: unitree_toolcall dataset not found")
            return
            
        # Extract tools from first item
        tools = unitree_toolcall_dataset[0]["tools"]
        
        # Process unitree_toolcall_dataset to extract call pairs
        weather_call_pairs = []
        other_call_pairs = []
        wo_obs_pairs = []
        other_wo_obs_pairs = []
        
        for data in unitree_toolcall_dataset:
            audio = data["audios"]
            messages = data["messages"]
            n = 0
            
            # Add audio paths to messages
            for message in messages:
                if "<audio>" in message["content"]:
                    audio_path = audio[n]
                    n += 1
                    message["audio"] = audio_path
            
            # Extract call pairs
            for i, message in enumerate(messages):
                if message["role"] == "function_call" and "天气" in message["content"]:
                    call_pairs = [
                        messages[i-1],
                        message,
                        messages[i+1],
                        messages[i+2]
                    ]
                    wo_obs_pairs.append([call_pairs[0], call_pairs[-1]])
                    weather_call_pairs.append(call_pairs)
                    break
                elif message["role"] == "function_call":
                    call_pairs = [
                        messages[i-1],
                        message,
                        messages[i+1],
                        messages[i+2]
                    ]
                    other_wo_obs_pairs.append([call_pairs[0], call_pairs[-1]])
                    other_call_pairs.append(call_pairs)
                    break
        
        # Load unitree dataset and prepare base conversations
        unitree_dataset = self.load_unitree_dataset()
        unitree_dataset_longer = unitree_dataset.filter(lambda x: len(x["messages"]) >= 17)
        unitree_dataset_removeaudio = unitree_dataset_longer.shuffle(seed=43).select(range(1500)).to_list()
        
        # Remove audio from base conversations
        for data in unitree_dataset_removeaudio:
            messages = data["messages"]
            data["audios"] = None
            for i, message in enumerate(messages):
                if message["role"] == "user" and "<audio>" in message["content"]:
                    message["content"] = message["content"].replace("<audio>", "")
            data["messages"] = messages
        
        # Process first 1000 items
        for data in unitree_dataset_removeaudio[:1000]:
            n = random.randint(1, 5)
            data["id"] = "unitree_mask_history"
            select_pairs = random.sample(wo_obs_pairs, n)
            select_pairs = deepcopy(select_pairs)
            data["audios"] = [select_pairs[i][0].pop("audio") for i in range(n)]
            for i in range(n):
                data["messages"].extend(select_pairs[i])
            
            call_pairs = random.sample(weather_call_pairs, 1)[0]
            call_pairs = deepcopy(call_pairs)
            data["audios"].append(call_pairs[0].pop("audio"))
            data["messages"].extend(call_pairs)
            data["tools"] = tools
        
        # Process remaining items
        for data in unitree_dataset_removeaudio[1000:]:
            n = random.randint(1, 3)
            data["id"] = "unitree_mask_history"
            select_pairs = random.sample(wo_obs_pairs, n)
            select_pairs = deepcopy(select_pairs)

            m = random.randint(1, 2)
            other_select_pairs = random.sample(other_wo_obs_pairs, m)
            other_select_pairs = deepcopy(other_select_pairs)
            select_pairs.extend(other_select_pairs)
            random.shuffle(select_pairs)
            data["audios"] = [select_pairs[i][0].pop("audio") for i in range(n+m)]
            for i in range(n+m):
                data["messages"].extend(select_pairs[i])
            
            if random.random() < 0.5: 
                call_pairs = random.sample(weather_call_pairs, 1)[0]
                call_pairs = deepcopy(call_pairs)
                data["audios"].append(call_pairs[0].pop("audio"))
                data["messages"].extend(call_pairs)
            else:
                call_pairs = random.sample(other_call_pairs, 1)[0]
                call_pairs = deepcopy(call_pairs)
                data["audios"].append(call_pairs[0].pop("audio"))
                data["messages"].extend(call_pairs)
            data["tools"] = tools
        
        # Save the generated data
        os.makedirs(os.path.dirname(self.config.unitree_mask_history_path), exist_ok=True)
        with open(self.config.unitree_mask_history_path, "w") as f:
            json.dump(unitree_dataset_removeaudio, f, ensure_ascii=False, indent=4)
        
        self.logger.info(f"Generated unitree_mask_history.json with {len(unitree_dataset_removeaudio)} samples")

    def process_all_datasets(self):
        """Main method to process and merge all datasets (original functionality + new features)"""
        # Generate unitree_mask_history.json if it doesn't exist
        self.generate_unitree_mask_history()
        
        # ===== New functionality from test.py =====
        print("Creating unitree dataset with appended tools...")
        unitree_dataset_append_tools = self.create_unitree_dataset_append_tools()
        unitree_dataset_append_tools = self.filter_dataset_features(unitree_dataset_append_tools, "unitree_dataset_append_tools")
        
        print("Creating unitree dataset with randomly removed audio...")
        unitree_dataset_randomremoveaudio = self.create_unitree_dataset_randomremoveaudio()
        unitree_dataset_randomremoveaudio = self.filter_dataset_features(unitree_dataset_randomremoveaudio, "unitree_dataset_randomremoveaudio")
        
        print("Creating unitree toolcall mixed dataset...")
        unitree_toolcall_mixed_dataset = self.create_unitree_toolcall_mixed_dataset()
        unitree_toolcall_mixed_dataset = self.filter_dataset_features(unitree_toolcall_mixed_dataset, "unitree_toolcall_mixed_dataset")
        
        # Merge the three new datasets
        new_datasets_to_merge = []
        for dataset in [unitree_dataset_append_tools, unitree_dataset_randomremoveaudio, unitree_toolcall_mixed_dataset]:
            if len(dataset) > 0:
                new_datasets_to_merge.append(dataset)
        
        if new_datasets_to_merge:
            merged_new_dataset = concatenate_datasets(new_datasets_to_merge).shuffle(seed=44)
            print(f"Merged new datasets size: {len(merged_new_dataset)}")
            merged_new_dataset = self.filter_dataset_features(merged_new_dataset, "merged_new_dataset")
        else:
            merged_new_dataset = Dataset.from_list([])
        
        # ===== Original functionality =====
        print("Loading Unitree toolcall datasets...")
        unitree_toolcall_dataset = self.load_json_dataset(self.config.unitree_toolcall_path)
        unitree_toolcall_dataset_mix = self.load_json_dataset(self.config.unitree_toolcall_mix_path) if os.path.exists(self.config.unitree_toolcall_mix_path) else Dataset.from_list([])
        unitree_toolcall_mask_history_dataset = self.load_json_dataset(self.config.unitree_mask_history_path)
        
        datasets_to_concat = [d for d in [unitree_toolcall_dataset, unitree_toolcall_dataset_mix, unitree_toolcall_mask_history_dataset] if len(d) > 0]
        if datasets_to_concat:
            unitree_toolcall_dataset = concatenate_datasets(datasets_to_concat)
        else:
            unitree_toolcall_dataset = Dataset.from_list([])
        unitree_toolcall_dataset = self.filter_dataset_features(unitree_toolcall_dataset, "unitree_toolcall_dataset")

        print("Loading Tulu dataset...")
        if os.path.exists(self.config.tulu_path):
            tulu_dataset = load_dataset(self.config.tulu_path)["train"]
            if "source" in tulu_dataset.column_names:
                tulu_dataset = tulu_dataset.remove_columns(["source"])
        else:
            self.logger.warning(f"Tulu dataset not found: {self.config.tulu_path}")
            tulu_dataset = Dataset.from_list([])
        tulu_dataset = self.filter_dataset_features(tulu_dataset, "tulu_dataset")
        gc.collect()

        print("Loading Glaive function calling dataset...")
        glaive_function_calling_dataset = self.load_json_dataset(self.config.glaive_func_path)
        if len(glaive_function_calling_dataset) > 0:
            glaive_function_calling_dataset = glaive_function_calling_dataset.map(
                self.convert_conv2messages, remove_columns=["conversations"] if "conversations" in glaive_function_calling_dataset.column_names else []
            )
            glaive_function_calling_dataset = glaive_function_calling_dataset.filter(lambda x: x.get("tools") is not None)
            glaive_function_calling_dataset_longer = glaive_function_calling_dataset.filter(lambda x: len(x["messages"]) > 10)
            glaive_function_calling_dataset_longer = glaive_function_calling_dataset_longer.map(self.remove_audio_tag)
            if "audios" in glaive_function_calling_dataset_longer.column_names:
                glaive_function_calling_dataset_longer = glaive_function_calling_dataset_longer.remove_columns(["audios"])
        else:
            glaive_function_calling_dataset_longer = Dataset.from_list([])
        glaive_function_calling_dataset = self.filter_dataset_features(glaive_function_calling_dataset, "glaive_function_calling_dataset")
        glaive_function_calling_dataset_longer = self.filter_dataset_features(glaive_function_calling_dataset_longer, "glaive_function_calling_dataset_longer")

        print("Loading XLAM dataset...") # text only
        xlam_dataset = self.load_json_dataset(self.config.xlam_path)
        if len(xlam_dataset) > 0:
            xlam_dataset = xlam_dataset.map(self.convert_xlam2messages, remove_columns=["query", "answers"] if "query" in xlam_dataset.column_names else [])
        xlam_dataset = self.filter_dataset_features(xlam_dataset, "xlam_dataset")

        # Get all tools
        print("Extracting tools...")
        tools = self.get_tools(glaive_function_calling_dataset)
        tools += self.get_tools(xlam_dataset)
        gc.collect()

        print("Loading LLaVA OneVision dataset...")
        if os.path.exists(self.config.llava_onevision_path):
            llava_onevision_dataset = load_dataset(self.config.llava_onevision_path)["train"]
            if "tools" in llava_onevision_dataset.column_names:
                llava_onevision_dataset = llava_onevision_dataset.remove_columns(["tools"])
            llava_onevision_dataset = llava_onevision_dataset.shuffle()
            print("llava_onevision_dataset", len(llava_onevision_dataset))
            llava_onevision_dataset = llava_onevision_dataset.filter(lambda x: len(x["images"]) > 0)
            print("llava_onevision_dataset after filter", len(llava_onevision_dataset))

            # Randomly select items matching Tulu dataset size
            total_items = len(llava_onevision_dataset)
            selection_size = len(tulu_dataset)
            if total_items > selection_size:
                random_indices = random.sample(range(total_items), selection_size)
                llava_onevision_dataset = llava_onevision_dataset.select(random_indices)
            else:
                print(f"Note: llava_onevision_dataset contains only {total_items} items, which is less than {selection_size}")
        else:
            llava_onevision_dataset = Dataset.from_list([])
        llava_onevision_dataset = self.filter_dataset_features(llava_onevision_dataset, "llava_onevision_dataset")
        gc.collect()

        print("Processing samples with tools...")
        tulu_samples_with_tools = self.add_random_tools_to_samples(
            tulu_dataset, tools, self.config.num_tulu_tools_samples
        )
        tulu_samples_with_tools = self.filter_dataset_features(tulu_samples_with_tools, "tulu_samples_with_tools")
        
        llava_samples_with_tools = self.add_random_tools_to_samples(
            llava_onevision_dataset, tools, self.config.num_llava_tools_samples
        )
        llava_samples_with_tools = self.filter_dataset_features(llava_samples_with_tools, "llava_samples_with_tools")

        print("Loading voice dataset...")
        if os.path.exists(self.config.voice_assistant_path):
            voice_dataset = load_dataset("json", data_files=self.config.voice_assistant_path)["train"]
            voice_samples_with_tools = self.add_random_tools_to_samples(
                voice_dataset, tools, self.config.num_voice_tools_samples
            )
        else:
            voice_dataset = Dataset.from_list([])
            voice_samples_with_tools = Dataset.from_list([])
        voice_dataset = self.filter_dataset_features(voice_dataset, "voice_dataset")
        voice_samples_with_tools = self.filter_dataset_features(voice_samples_with_tools, "voice_samples_with_tools")

        print("Loading system prompt dataset...")
        if os.path.exists(self.config.system_prompt_path):
            system_prompt_dataset = load_dataset(self.config.system_prompt_path)["train"]
            system_prompt_dataset = system_prompt_dataset.map(self.revert_role_content, remove_columns=["messages"] if "messages" in system_prompt_dataset.column_names else [])
            system_prompt_dataset = system_prompt_dataset.rename_column("messages_new", "messages")
            system_prompt_samples_with_tools = self.add_random_tools_to_samples(
                system_prompt_dataset, tools, self.config.num_system_prompt_samples
            )
        else:
            system_prompt_dataset = Dataset.from_list([])
            system_prompt_samples_with_tools = Dataset.from_list([])
        system_prompt_dataset = self.filter_dataset_features(system_prompt_dataset, "system_prompt_dataset")
        system_prompt_samples_with_tools = self.filter_dataset_features(system_prompt_samples_with_tools, "system_prompt_samples_with_tools")

        print("Loading Unitree dataset...")
        unitree_dataset = self.load_unitree_dataset()
        unitree_samples_with_tools = self.add_random_tools_to_samples(
            unitree_dataset, tools, self.config.num_unitree_tools_samples
        )
        unitree_dataset = self.filter_dataset_features(unitree_dataset, "unitree_dataset")
        unitree_samples_with_tools = self.filter_dataset_features(unitree_samples_with_tools, "unitree_samples_with_tools")

        # Extract system prompts
        system_prompts = []
        for example in system_prompt_dataset:
            if "messages" in example and len(example["messages"]) > 0 and example["messages"][0]["role"] == "system":
                system_prompts.append(example["messages"][0])

        print("Processing datasets with system prompts...")
        tulu_selected = tulu_samples_with_tools.shuffle().select(range(min(self.config.num_selected_with_system, len(tulu_samples_with_tools))))
        tulu_with_system = tulu_selected.map(
            lambda example: self.add_system_prompt(example, system_prompts)
        )
        tulu_with_system = self.filter_dataset_features(tulu_with_system, "tulu_with_system")

        llava_selected = llava_samples_with_tools.shuffle().select(range(min(self.config.num_selected_with_system, len(llava_samples_with_tools))))
        llava_with_system = llava_selected.map(
            lambda example: self.add_system_prompt(example, system_prompts)
        )
        llava_with_system = self.filter_dataset_features(llava_with_system, "llava_with_system")

        voice_selected = voice_samples_with_tools.shuffle().select(range(min(self.config.num_selected_with_system, len(voice_samples_with_tools))))
        voice_with_system = voice_selected.map(
            lambda example: self.add_system_prompt(example, system_prompts)
        )
        voice_with_system = self.filter_dataset_features(voice_with_system, "voice_with_system")

        print("Loading augmented dataset...")
        if os.path.exists(self.config.aug_dataset_path):
            with open(self.config.aug_dataset_path, "r") as f:
                aug_dataset = json.load(f)
            aug_messages = []
            for example in aug_dataset:
                aug_messages.append({"content": example, "role": "system"})
        else:
            aug_messages = []

        # Process datasets with augmented system prompts
        glaive_dataset_with_aug = glaive_function_calling_dataset.shuffle().select(range(min(self.config.num_glaive_aug_samples, len(glaive_function_calling_dataset))))
        glaive_dataset_with_aug = glaive_dataset_with_aug.map(
            lambda example: self.add_system_prompt(example, aug_messages)
        )
        glaive_dataset_with_aug = self.filter_dataset_features(glaive_dataset_with_aug, "glaive_dataset_with_aug")

        xlam_dataset_with_aug = xlam_dataset.shuffle().select(range(min(self.config.num_xlam_aug_samples, len(xlam_dataset))))
        xlam_dataset_with_aug = xlam_dataset_with_aug.map(
            lambda example: self.add_system_prompt(example, aug_messages)
        )
        xlam_dataset_with_aug = self.filter_dataset_features(xlam_dataset_with_aug, "xlam_dataset_with_aug")

        llave_onevision_dataset_with_aug = llava_onevision_dataset.shuffle().select(range(min(self.config.num_llava_aug_samples, len(llava_onevision_dataset))))
        llave_onevision_dataset_with_aug = llave_onevision_dataset_with_aug.map(
            lambda example: self.add_system_prompt(example, aug_messages)
        )
        llave_onevision_dataset_with_aug = self.filter_dataset_features(llave_onevision_dataset_with_aug, "llave_onevision_dataset_with_aug")

        voice_dataset_with_aug = voice_dataset.shuffle().select(range(min(self.config.num_voice_aug_samples, len(voice_dataset))))
        voice_dataset_with_aug = voice_dataset_with_aug.map(
            lambda example: self.add_system_prompt(example, aug_messages)
        )
        voice_dataset_with_aug = self.filter_dataset_features(voice_dataset_with_aug, "voice_dataset_with_aug")

        print("Loading tools with audio datasets...")
        tools_with_audio_path = [
            # self.config.glaive_toolcall_en_demo_path, # remove en
            self.config.glaive_toolcall_zh_demo_path
        ]
        available_paths = [p for p in tools_with_audio_path if os.path.exists(p)]
        if available_paths:
            tools_with_audio_datasets = load_dataset("json", data_files=available_paths)["train"]
            tools_with_audio_datasets = tools_with_audio_datasets.map(
                self.convert_conv2messages, remove_columns=["conversations"] if "conversations" in tools_with_audio_datasets.column_names else []
            )
        else:
            tools_with_audio_datasets = Dataset.from_list([])
        tools_with_audio_datasets = self.filter_dataset_features(tools_with_audio_datasets, "tools_with_audio_datasets")
        
        # [id, tools, images, audios, messages]
        print("Merging all datasets...")
        datasets_to_merge = []
        for dataset in [
            tulu_dataset, tulu_samples_with_tools, tulu_with_system,
            glaive_function_calling_dataset, glaive_function_calling_dataset_longer,
            xlam_dataset,
            llava_onevision_dataset, llava_samples_with_tools, llava_with_system,
            voice_dataset, voice_samples_with_tools, voice_with_system,
            system_prompt_dataset, system_prompt_samples_with_tools,
            unitree_dataset, unitree_samples_with_tools,
            glaive_dataset_with_aug, xlam_dataset_with_aug,
            llave_onevision_dataset_with_aug, voice_dataset_with_aug,
            tools_with_audio_datasets,
            unitree_toolcall_dataset,
            merged_new_dataset
        ]:
            if len(dataset) > 0:
                datasets_to_merge.append(dataset)
        
        if datasets_to_merge:
            merged_dataset = concatenate_datasets(datasets_to_merge).shuffle()
        else:
            merged_dataset = Dataset.from_list([])

        print(f"Total merged dataset size: {len(merged_dataset)}")
        print(f"Saving to: {self.config.output_path}")
        self.save_llava_dataset(merged_dataset, self.config.output_path)
        print("Dataset processing completed!")

    def process_random_combine(self):
        """Process random combination of datasets"""
        self.set_seed()
        
        print("Loading datasets for random combination...")
        unitree_dataset = self.load_unitree_dataset()
        fight_qa_dataset = self.load_fight_qa_dataset()
        voiceassistant_dataset = self.load_voiceassistant_dataset()

        # Convert datasets to lists for easier manipulation
        unitree_data = unitree_dataset.to_list()
        fight_qa_data = fight_qa_dataset.to_list()
        voiceassistant_data = voiceassistant_dataset.to_list()
        
        # Randomly select samples
        selected_unitree = random.sample(unitree_data, min(self.config.num_unitree_samples, len(unitree_data)))
        selected_fight_qa = random.sample(fight_qa_data, min(self.config.num_fight_qa_samples, len(fight_qa_data)))
        
        # Combine the selected conversations
        selected_conversations = selected_unitree + selected_fight_qa
        
        # Process each selected conversation
        combined_data = []
        for conversation in selected_conversations:
            # Randomly select voice assistant conversations
            num_va_convs = random.randint(self.config.min_va_conversations, self.config.max_va_conversations)
            selected_va_convs = random.sample(voiceassistant_data, min(num_va_convs, len(voiceassistant_data)))
            
            # Combine conversations
            combined_conversation = self.combine_conversations(conversation, selected_va_convs)
            combined_data.append(combined_conversation)
        
        # Save the combined dataset
        combined_data = combined_data + selected_unitree + selected_fight_qa
        
        # Add required fields
        for i in range(len(combined_data)):
            combined_data[i]["tools"] = None
            combined_data[i]["images"] = None
        random.shuffle(combined_data)
        
        # Find special item and duplicate
        for combined_data_item in combined_data:
            if len(combined_data_item["messages"]) > 1 and combined_data_item["messages"][1]["content"] == "<audio>膝撞":
                break
        combined_data = combined_data + [dict(combined_data_item)] * self.config.duplicate_count
        
        print(f"Total combined data: {len(combined_data)}")
        
        # Mix with base dataset if available
        if os.path.exists(self.config.mix_base_dataset_path):
            base_dataset = load_dataset(self.config.mix_base_dataset_path, split="train")
            combined_dataset = Dataset.from_list(combined_data)
            combined_dataset = concatenate_datasets([base_dataset, combined_dataset]).shuffle()
        else:
            combined_dataset = Dataset.from_list(combined_data)
        
        output_path = self.config.output_path.replace("_v2_1", "_random_combine")
        self.save_llava_dataset(combined_dataset, output_path)
        print(f"Random combine processing completed! Saved to: {output_path}")

    def process_qwen_vl_mix(self):
        """Process Qwen VL mixing"""
        print("Loading datasets for Qwen VL mixing...")
        
        # Load ShareRobot dataset
        if os.path.exists(self.config.sharerobot_path):
            sharerobot_dataset = load_dataset("json", data_files=self.config.sharerobot_path)["train"]
            sharerobot_dataset = sharerobot_dataset.remove_columns(['task', 'selected_step', 'image_path', 'instruction', 'affordance', 'meta_data', 'trajectory'])
            sharerobot_dataset = sharerobot_dataset.rename_column("image", "images")
            sharerobot_dataset = sharerobot_dataset.map(self.convert_sharerobot, remove_columns=["conversations"])
            sharerobot_dataset = sharerobot_dataset.filter(lambda x: len(x["images"]) == len(re.findall("<image>", x["messages"][0]["content"])))
            # Join the path to images path ShareRobot/planning prefix
            sharerobot_dataset = sharerobot_dataset.map(lambda x: {"images": [os.path.join("ShareRobot/planning", image) for image in x["images"]]})
            print("ShareRobot dataset loaded:", len(sharerobot_dataset))
        else:
            sharerobot_dataset = Dataset.from_list([])

        # Load LLaVA OneVision dataset
        if os.path.exists(self.config.llava_onevision_path):
            llava_onevision_dataset = load_dataset(self.config.llava_onevision_path)["train"]
            llava_onevision_dataset = llava_onevision_dataset.filter(lambda x: len(x["images"]) > 0)
            if "tools" in llava_onevision_dataset.column_names:
                llava_onevision_dataset = llava_onevision_dataset.remove_columns(["tools"])
            print("LLaVA OneVision dataset loaded:", len(llava_onevision_dataset))
        else:
            llava_onevision_dataset = Dataset.from_list([])

        # Load pointing dataset
        if os.path.exists(self.config.pixmo_points_path):
            pointing_dataset = load_dataset("json", data_files=self.config.pixmo_points_path)["train"]
            pointing_dataset = pointing_dataset.map(self.process_pointing)
        else:
            pointing_dataset = Dataset.from_list([])

        # Load RoboSpatial dataset
        if os.path.exists(self.config.robospatial_det_path):
            robospatial_dataset = load_dataset("json", data_files=[self.config.robospatial_det_path])["train"]
            robospatial_dataset = robospatial_dataset.map(self.process_robospatial)
        else:
            robospatial_dataset = Dataset.from_list([])

        # Load RoboSpatial dataset v2
        if os.path.exists(self.config.robospatial_data_dir):
            qa_files = [os.path.join(self.config.robospatial_data_dir, f) for f in os.listdir(self.config.robospatial_data_dir) if f.endswith(".json") and f.startswith("qa_")]
            if qa_files:
                robospatial_dataset_v2 = load_dataset("json", data_files=qa_files)["train"]
                robospatial_dataset_v2 = robospatial_dataset_v2.map(self.process_robospatial_v2, remove_columns=["image_path", "question", "answer"])
                robospatial_dataset_v2 = robospatial_dataset_v2.shuffle(seed=42).select(range(int(len(robospatial_dataset_v2) * self.config.robospatial_subset_ratio)))
                print("RoboSpatial dataset v2 loaded:", len(robospatial_dataset_v2))
            else:
                robospatial_dataset_v2 = Dataset.from_list([])
        else:
            robospatial_dataset_v2 = Dataset.from_list([])

        gc.collect()

        # Load text dataset
        if os.path.exists(self.config.tulu_path):
            text_dataset = load_dataset(self.config.tulu_path)["train"]
            text_dataset = text_dataset.shuffle(seed=42).select(range(int(len(text_dataset) * self.config.tulu_subset_ratio)))
        else:
            text_dataset = Dataset.from_list([])

        # Merge datasets
        datasets_to_merge = []
        for dataset in [sharerobot_dataset, llava_onevision_dataset, robospatial_dataset, robospatial_dataset_v2, pointing_dataset, text_dataset]:
            if len(dataset) > 0:
                datasets_to_merge.append(dataset)
        
        if datasets_to_merge:
            merged_dataset = concatenate_datasets(datasets_to_merge).shuffle(seed=42)
        else:
            merged_dataset = Dataset.from_list([])
        
        print(f"Total merged dataset size: {len(merged_dataset)}")
        
        output_path = self.config.output_path.replace("_v2_1", "_qwen_vl_mixed_v2")
        self.save_llava_dataset(merged_dataset, output_path)
        print(f"Qwen VL mixing completed! Saved to: {output_path}")

    # ===== Additional processing methods from test.py =====
    def create_unitree_dataset_append_tools(self):
        """Create unitree dataset with appended tools (from test.py)"""
        # Load required datasets
        unitree_toolcall_dataset = self.load_json_dataset(self.config.unitree_toolcall_path)
        glaive_toolcall_dataset = self.load_json_dataset(self.config.glaive_func_path)
        
        if len(unitree_toolcall_dataset) == 0 or len(glaive_toolcall_dataset) == 0:
            self.logger.warning("Required datasets not found for unitree_dataset_append_tools")
            return Dataset.from_list([])
        
        # Process unitree_toolcall_dataset to extract tools and call pairs
        tools = unitree_toolcall_dataset[0]["tools"]
        all_call_pairs = []
        
        for data in unitree_toolcall_dataset:
            audio = data["audios"]
            messages = data["messages"]
            n = 0
            for message in messages:
                if "<audio>" in message["content"]:
                    audio_path = audio[n]
                    n += 1
                    message["audio"] = audio_path
            
            for i, message in enumerate(messages):
                if message["role"] == "function_call":
                    call_pairs = [
                        messages[i-1],
                        message,
                        messages[i+1],
                        messages[i+2]
                    ]
                    all_call_pairs.append(call_pairs)
                    break
        
        all_call_pairs = all_call_pairs[::-1]
        
        # Process glaive dataset to extract tools and call pairs
        glaive_toolcall_dataset_filtered = [d for d in glaive_toolcall_dataset if len(json.loads(d["tools"])) > 0 and "play_music" not in d["tools"]]
        glaive_all_call_pairs = []
        glaive_tools = []
        
        for data in glaive_toolcall_dataset_filtered:
            messages = data["conversations"]
            for i, message in enumerate(messages):
                if message["from"] == "function_call":
                    call_pairs = [
                        messages[i-1],
                        message,
                        messages[i+1],
                        messages[i+2]
                    ]
                    key_map = {
                        "human": "user",
                        "gpt": "assistant",
                        "function_call": "function_call",
                        "observation": "observation"
                    }
                    processed_messages = []
                    for msg in call_pairs:
                        processed_messages.append({
                            "content": msg["value"],
                            "role": key_map[msg["from"]]
                        })
                    glaive_all_call_pairs.append({"messages": processed_messages, "tools": data["tools"]})
                    glaive_tools.append(data["tools"])
                    break
        
        random.shuffle(glaive_all_call_pairs)
        
        # Load unitree dataset and create append_tools version
        unitree_dataset = self.load_unitree_dataset()
        unitree_dataset_append_tools = unitree_dataset.shuffle(seed=42).select(range(min(1000, len(unitree_dataset)))).to_list()
        
        # Process first 500 items with unitree toolcalls
        for i, data in enumerate(unitree_dataset_append_tools[:min(500, len(unitree_dataset_append_tools))]):
            if i < len(all_call_pairs):
                toolcall = [dict(_) for _ in all_call_pairs[i % len(all_call_pairs)]]
                audio = toolcall[0].pop("audio")
                data["messages"].extend(toolcall)
                data["tools"] = tools
                data["audios"] = data["audios"] + [audio]
        
        # Process remaining items with glaive toolcalls
        for i, data in enumerate(unitree_dataset_append_tools[500:]):
            if i < len(glaive_all_call_pairs):
                toolcall = dict(glaive_all_call_pairs[i % len(glaive_all_call_pairs)])
                data["messages"].extend(toolcall["messages"])
                data["tools"] = toolcall["tools"]
        
        return Dataset.from_list(unitree_dataset_append_tools)
    
    def create_unitree_dataset_randomremoveaudio(self):
        """Create unitree dataset with randomly removed audio (from test.py)"""
        unitree_dataset = self.load_unitree_dataset()
        if len(unitree_dataset) == 0:
            return Dataset.from_list([])
            
        unitree_dataset_longer = unitree_dataset.filter(lambda x: len(x["messages"]) >= 17)
        
        # First 1000: remove audio from first 5-8 messages
        unitree_dataset_removeaudio = unitree_dataset_longer.shuffle(seed=43).select(range(min(1000, len(unitree_dataset_longer)))).to_list()
        
        for i, data in enumerate(unitree_dataset_removeaudio):
            audio_num = random.randint(5, 8)
            n = 0
            for j in range(audio_num):
                if 2*j+1 < len(data["messages"]) and "<audio>" in data["messages"][2*j+1]["content"]:
                    n += 1
                    data["messages"][2*j+1]["content"] = data["messages"][2*j+1]["content"].replace("<audio>", "")
            data["audios"] = data["audios"][n:]
        
        # Next 1000: randomly remove audio
        if len(unitree_dataset_longer) > 1000:
            unitree_dataset_randomremoveaudio = unitree_dataset_longer.select(range(1000, min(2000, len(unitree_dataset_longer)))).to_list()
            
            for i, data in enumerate(unitree_dataset_randomremoveaudio):
                messages = data["messages"]
                audios = data["audios"]
                new_messages = []
                new_audios = []
                audio_idx = 0
                
                for message in messages:
                    if message["role"] != "user":
                        new_messages.append(message)
                        continue

                    if "<audio>" in message["content"] and random.random() < 0.5:
                        new_messages.append(message)
                        if audio_idx < len(audios):
                            new_audios.append(audios[audio_idx])
                    else:
                        message_copy = message.copy()
                        message_copy["content"] = message["content"].replace("<audio>", "")
                        new_messages.append(message_copy)
                    
                    if "<audio>" in message["content"]:
                        audio_idx += 1
                
                data["messages"] = new_messages
                data["audios"] = new_audios
            
            combined_data = unitree_dataset_randomremoveaudio + unitree_dataset_removeaudio
        else:
            combined_data = unitree_dataset_removeaudio
        
        return Dataset.from_list(combined_data)
    
    def create_unitree_toolcall_mixed_dataset(self):
        """Create unitree toolcall dataset mixed with glaive tools (from test.py)"""
        unitree_toolcall_dataset = self.load_json_dataset(self.config.unitree_toolcall_path)
        glaive_toolcall_dataset = self.load_json_dataset(self.config.glaive_func_path)
        
        if len(unitree_toolcall_dataset) == 0 or len(glaive_toolcall_dataset) == 0:
            self.logger.warning("Required datasets not found for unitree_toolcall_mixed_dataset")
            return Dataset.from_list([])
        
        # Extract glaive tools
        glaive_toolcall_dataset_filtered = [d for d in glaive_toolcall_dataset if len(json.loads(d["tools"])) > 0 and "play_music" not in d["tools"]]
        glaive_tools = [data["tools"] for data in glaive_toolcall_dataset_filtered]
        
        # Mix tools into unitree toolcall dataset
        glaive_select_parsed = json.loads(data["tools"]) # list
        if isinstance(glaive_select_parsed, str):
            glaive_select_parsed = [glaive_select_parsed]
        for i, data in enumerate(unitree_toolcall_dataset):
            n = random.randint(1, min(5, len(glaive_tools)))
            glaive_select = random.sample(glaive_tools, n)
            
            for gs in glaive_select:
                gs = json.loads(gs)
                if isinstance(gs, str):
                    gs = [gs]
                glaive_select_parsed += gs
            random.shuffle(glaive_select_parsed)
            data["tools"] = json.dumps(glaive_select_parsed, ensure_ascii=False)
        
        return Dataset.from_list(unitree_toolcall_dataset)


def main():
    parser = argparse.ArgumentParser(description='Process and mix multiple datasets')
    parser.add_argument('--mode', type=str, choices=['full', 'random_combine', 'qwen_vl_mix'], 
                       default='full', help='Processing mode: full (original), random_combine, or qwen_vl_mix')
    parser.add_argument('--config-file', type=str, 
                       help='Path to configuration file (JSON format)')
    parser.add_argument('--output-path', type=str,
                       help='Override output path')
    parser.add_argument('--data-root', type=str,
                       help='Override data root path')
    parser.add_argument('--llama-factory-root', type=str,
                       help='Override LLaMA Factory root path')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize config manager
    config_manager = ConfigManager(config_file=args.config_file)
    
    # Prepare overrides
    overrides = {}
    if args.output_path:
        overrides['output_path'] = args.output_path
    if args.data_root:
        overrides['data_root'] = args.data_root
    if args.llama_factory_root:
        overrides['llama_factory_root'] = args.llama_factory_root
    
    # Get configuration
    config = config_manager.get_data_mix_config(**overrides)
    
    print(f"Processing mode: {args.mode}")
    print("Configuration loaded:")
    print(f"  Data root: {config.data_root}")
    print(f"  LLaMA Factory root: {config.llama_factory_root}")
    print(f"  Output path: {config.output_path}")
    
    # Initialize and run data mixer
    mixer = DataMixer(config)
    
    if args.mode == 'full':
        mixer.process_all_datasets()
    elif args.mode == 'random_combine':
        mixer.process_random_combine()
    elif args.mode == 'qwen_vl_mix':
        mixer.process_qwen_vl_mix()


if __name__ == "__main__":
    main() 