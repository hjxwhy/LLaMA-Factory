import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class DatasetConfig:
    """Configuration for dataset parsing"""
    input_path: str
    output_path: str
    batch_size: int = 1000
    max_workers: int = 8
    skip_existing: bool = False
    

@dataclass
class VoiceAssistantConfig(DatasetConfig):
    save_audio_dir: str = "audio"
    

@dataclass
class LLaVAConfig(DatasetConfig):
    image_format: str = "JPEG"
    create_combined_file: bool = True
    

@dataclass
class PixmoConfig(DatasetConfig):
    name: str = "pixmo-points"
    # Download settings
    max_concurrent: int = 100
    timeout: int = 30
    max_retries: int = 3
    verify_hash: bool = True
    debug_mode: bool = False
    save_failed_hash_samples: bool = True
    connection_limit: int = 100
    connection_limit_per_host: int = 30
    queue_size: int = 1000
    chunk_size: int = 8192
    
    # Image processing
    point_scale: int = 100
    
    # Output settings
    create_visualization: bool = False
    output_jsonl: bool = True
    

@dataclass
class DataMixConfig:
    """Configuration for data mixing and merging operations"""
    # Base paths
    data_root: str
    llama_factory_root: str
    
    # Input dataset paths
    tulu_path: str
    glaive_func_path: str
    xlam_path: str
    llava_onevision_path: str
    voice_assistant_path: str
    system_prompt_path: str
    aug_dataset_path: str
    
    # Unitree specific paths
    unitree_audio_user_path: str
    unitree_g1_instruct_path: str
    unitree_toolcall_path: str
    unitree_toolcall_mix_path: str
    unitree_mask_history_path: str
    unitree_fight_qa_path: str
    
    # Tools with audio paths
    glaive_toolcall_en_demo_path: str
    glaive_toolcall_zh_demo_path: str
    
    # Random combine specific paths
    unitree_combined_conversations_path: str
    mix_base_dataset_path: str
    
    # Qwen VL mix specific paths
    sharerobot_path: str
    pixmo_points_path: str
    robospatial_data_dir: str
    robospatial_det_path: str
    
    # Output path
    output_path: str
    
    # Processing parameters
    num_tulu_tools_samples: int = 10000
    num_llava_tools_samples: int = 10000
    num_voice_tools_samples: int = 10000
    num_system_prompt_samples: int = 1000
    num_unitree_tools_samples: int = 1000
    num_glaive_aug_samples: int = 5000
    num_xlam_aug_samples: int = 1000
    num_llava_aug_samples: int = 1000
    num_voice_aug_samples: int = 1000
    num_selected_with_system: int = 1000
    
    # Random combine parameters
    num_unitree_samples: int = 200
    num_fight_qa_samples: int = 20
    min_va_conversations: int = 3
    max_va_conversations: int = 5
    duplicate_count: int = 100
    
    # Qwen VL mix parameters
    tulu_subset_ratio: float = 0.5
    robospatial_subset_ratio: float = 0.1


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.base_config = self._load_base_config()
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from file or environment"""
        config = {
            "data_root": os.getenv("DATA_ROOT", "/DATA"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "temp_dir": os.getenv("TEMP_DIR", "/tmp"),
        }
        
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
            config.update(file_config)
        
        return config
    
    def _flatten_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a nested config dictionary."""
        flat_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_config[sub_key] = sub_value
            else:
                flat_config[key] = value
        return flat_config

    def get_dataset_config(self, dataset_type: str, **overrides) -> DatasetConfig:
        """Get configuration for specific dataset type"""
        base_paths = {
            "voiceassistant": {
                "input_path": f"{self.base_config['data_root']}/disk0/data/VoiceAssistant-400K",
                "output_path": f"{self.base_config['data_root']}/disk1/data/VoiceAssistant-400K-Parsed"
            },
            "blip3": {
                "input_path": f"{self.base_config['data_root']}/disk0/blip3-grounding-50m/data",
                "output_path": f"{self.base_config['data_root']}/disk1/blip3-grounding-50m-parsed"
            },
            "llava_onevision": {
                "input_path": f"{self.base_config['data_root']}/disk0/data/LLaVA-OneVision-Data",
                "output_path": f"{self.base_config['data_root']}/disk1/data/LLaVA-OneVision-Data-Parsed"
            },
            "llava_video": {
                "input_path": f"{self.base_config['data_root']}/disk0/data/LLaVA-Video-178K",
                "output_path": f"{self.base_config['data_root']}/disk1/data/LLaVA-Video-178K-Parsed"
            },
            "pixmo": {
                "input_path": "/localfolder/data/pixmo-points",
                "output_path": f"{self.base_config['data_root']}/disk0/pixmo-points"
            }
        }
        
        if dataset_type not in base_paths:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        config_data = base_paths[dataset_type].copy()
        config_data.update(overrides)
        
        # Return specific config type based on dataset
        if dataset_type == "voiceassistant":
            return VoiceAssistantConfig(**config_data)
        elif dataset_type in ["llava_onevision", "llava_video"]:
            return LLaVAConfig(**config_data)
        elif dataset_type == "pixmo":
            return PixmoConfig(**config_data)
        else:
            return DatasetConfig(**config_data)
    
    def save_config(self, config: DatasetConfig, filepath: str):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(config), f, indent=2)

    def get_data_mix_config(self, **overrides) -> DataMixConfig:
        """Get configuration for data mixing operations"""
        # Load from the config file if it exists, otherwise use defaults
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
            
            # Flatten the nested structure from the JSON file
            # e.g., "paths": {"data_root": ...} becomes "data_root": ...
            loaded_config = self._flatten_config(file_config)
        else:
            loaded_config = {}

        # Prioritize environment variables, then file config, then defaults
        data_root = os.getenv("DATA_ROOT", loaded_config.get("data_root", "/localfolder/data"))
        llama_factory_root = os.getenv("LLAMA_FACTORY_ROOT", loaded_config.get("llama_factory_root", "/localfolder/code/LLaMA-Factory"))

        # Base config with dynamically constructed paths
        base_config = {
            "data_root": data_root,
            "llama_factory_root": llama_factory_root,
            
            # Input dataset paths - construct from base paths
            "tulu_path": f"{data_root}/tulu-3-sft-olmo-2-mixture-0225",
            "glaive_func_path": f"{llama_factory_root}/data/data/glaive_toolcall/glaive_toolcall_conversation_v2_noised.json",
            "xlam_path": f"{data_root}/xlam-function-calling-60k/xlam_function_calling_60k.json",
            "llava_onevision_path": f"{llama_factory_root}/data/data/llava-onevision-data",
            "voice_assistant_path": f"{llama_factory_root}/data/data/VoiceAssistant-400K-Parsed/voiceassistant.jsonl",
            "system_prompt_path": f"{data_root}/System-Prompt-Instruction-Real-world-Implementation-Training-set",
            "aug_dataset_path": f"{llama_factory_root}/data/data/complex_system_prompts/all_aug_system_prompt.json",
            
            # Unitree specific paths
            "unitree_audio_user_path": f"{llama_factory_root}/data/data/unitree_train/common_prompts/audio_user",
            "unitree_g1_instruct_path": f"{llama_factory_root}/data/data/unitree_train/common_prompts/g1_instruct_data/conversation_v2.json",
            "unitree_toolcall_path": f"{llama_factory_root}/data/data/glaive_toolcall/uniree_toolcall_conversation_v2.json",
            "unitree_toolcall_mix_path": f"{llama_factory_root}/data/data/glaive_toolcall/toolcall_mix",
            "unitree_mask_history_path": f"{llama_factory_root}/data/data/glaive_toolcall/unitree_mask_history.json",
            "unitree_fight_qa_path": f"{llama_factory_root}/data/data/unitree_train/fight_qa/combined_conversations.json",
            
            # Tools with audio paths
            "glaive_toolcall_en_demo_path": f"{llama_factory_root}/data/data/glaive_toolcall/glaive_toolcall_en_demo_conversation_v2.json",
            "glaive_toolcall_zh_demo_path": f"{llama_factory_root}/data/data/glaive_toolcall/glaive_toolcall_zh_demo_conversation_v2.json",
            
            # Random combine specific paths
            "unitree_combined_conversations_path": f"{llama_factory_root}/data/data/unitree_train/combined_conversations.json",
            "mix_base_dataset_path": f"{llama_factory_root}/data/data/mix_text_image_audio_tools_200w_unitreetools",
            
            # Qwen VL mix specific paths
            "sharerobot_path": f"{llama_factory_root}/data/data/ShareRobot/sharerobot_mixed.jsonl",
            "pixmo_points_path": f"{llama_factory_root}/data/data/pixmo-points/pixmo-points-llama-factory.jsonl",
            "robospatial_data_dir": f"{llama_factory_root}/data/data/EmbodiedScan/robospatial",
            "robospatial_det_path": f"{llama_factory_root}/data/data/EmbodiedScan/robospatial/train_data_chunk_0_obj_det.json",
            
            # Output path
            "output_path": f"{llama_factory_root}/data/data/mix_text_image_audio_tools_200w_unitreetools_v2_1"
        }

        # Update base_config with values from the flattened file config
        # This allows users to override any path or parameter via the JSON file
        base_config.update(loaded_config)
        
        # Finally, apply any command-line overrides
        base_config.update(overrides)
        
        return DataMixConfig(**base_config)


def setup_argparse(parser_name: str) -> argparse.ArgumentParser:
    """Setup common argument parser"""
    parser = argparse.ArgumentParser(description=f'{parser_name} dataset parser')
    parser.add_argument('--input-path', type=str, help='Input dataset path')
    parser.add_argument('--output-path', type=str, help='Output directory path') 
    parser.add_argument('--config-file', type=str, help='Configuration file path')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of workers')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing files')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    return parser 