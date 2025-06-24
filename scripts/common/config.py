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
    skip_existing: bool = True
    

@dataclass
class VoiceAssistantConfig(DatasetConfig):
    save_audio_dir: str = "audio"
    

@dataclass
class LLaVAConfig(DatasetConfig):
    image_format: str = "JPEG"
    create_combined_file: bool = True
    

@dataclass
class PixmoConfig(DatasetConfig):
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