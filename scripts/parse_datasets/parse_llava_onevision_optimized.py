#!/usr/bin/env python3
"""
Optimized LLaVA OneVision Dataset Parser

This script processes the LLaVA OneVision dataset by extracting images and conversations
from multiple configurations and converting them to a standardized format.

Features:
- Configurable input/output paths
- Resume capability for interrupted runs
- Comprehensive error handling and logging
- Progress tracking per configuration
- Image format validation and conversion
- Memory-efficient processing
"""

import os
import json
from datasets import get_dataset_config_names
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from PIL import Image

# Import common utilities
try:
    from ..common import (
        ConfigManager, LLaVAConfig, setup_argparse, 
        setup_logging, DatasetProcessor, load_dataset_safe,
        save_json_data, safe_makedirs
    )
except ImportError:
    # Fallback for when running without common module
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from common import (
        ConfigManager, LLaVAConfig, setup_argparse,
        setup_logging, DatasetProcessor, load_dataset_safe,
        save_json_data, safe_makedirs
    )


class LLaVAOneVisionParser(DatasetProcessor):
    """LLaVA OneVision dataset parser with enhanced functionality"""
    
    def __init__(self, config: LLaVAConfig):
        super().__init__(config)
        self.config = config
        self.processed_configs = set()
        self.final_data = []
        self._load_existing_progress()
    
    def _load_existing_progress(self):
        """Load existing progress to support resume functionality"""
        progress_file = os.path.join(self.config.output_path, "progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    self.processed_configs = set(progress.get('processed_configs', []))
                    self.log_progress(f"Loaded {len(self.processed_configs)} processed configs from previous run")
            except Exception as e:
                self.log_progress(f"Could not load progress file: {str(e)}", "warning")
    
    def _save_progress(self):
        """Save current progress"""
        progress_file = os.path.join(self.config.output_path, "progress.json")
        try:
            progress = {
                'processed_configs': list(self.processed_configs),
                'total_processed': self.processed_count,
                'total_errors': self.error_count
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.log_progress(f"Could not save progress: {str(e)}", "warning")
    
    def get_dataset_configs(self) -> List[str]:
        """
        Get available dataset configurations
        
        Returns:
            List of configuration names
        """
        try:
            configs = get_dataset_config_names(self.config.input_path)
            self.log_progress(f"Found configurations: {configs}")
            return configs
        except Exception as e:
            self.log_progress(f"Error getting dataset configs: {str(e)}", "error")
            return []
    
    def save_image_safe(self, image: Image.Image, image_path: str, item_id: str) -> bool:
        """
        Safely save image with format conversion and validation
        
        Args:
            image: PIL Image object
            image_path: Output path for image
            item_id: Item identifier for error reporting
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if needed
            safe_makedirs(os.path.dirname(image_path))
            
            # Convert image mode if necessary
            if image.mode in ("RGBA", "P", "LA"):
                self.log_progress(f"Converting image mode from {image.mode} to RGB for item {item_id}")
                image = image.convert("RGB")
            
            # Save image
            image.save(image_path, format=self.config.image_format)
            return True
            
        except Exception as e:
            self.log_progress(f"Failed to save image for item {item_id}: {str(e)}", "error")
            return False
    
    def process_config_item(self, data: Dict[str, Any], config_name: str, 
                           image_folder: str) -> Optional[Dict[str, Any]]:
        """
        Process a single data item from a configuration
        
        Args:
            data: Raw data item
            config_name: Configuration name
            image_folder: Folder for saving images
            
        Returns:
            Processed item or None if failed
        """
        try:
            item_id = data.get("id")
            if not item_id:
                self.log_progress("Item missing ID field", "warning")
                return None
            
            # Initialize result
            result = {
                "id": item_id,
                "config": config_name
            }
            
            # Handle image if present
            if data.get("image") is not None:
                image_filename = f"{item_id}.jpg"
                result["image"] = f"{config_name}/{image_filename}"
                image_path = os.path.join(self.config.output_path, result["image"])
                
                if not self.save_image_safe(data["image"], image_path, item_id):
                    self.error_count += 1
                    return None
            
            # Copy conversation data
            conversations = data.get("conversations")
            if conversations is None:
                self.log_progress(f"Item {item_id} missing conversations", "warning")
                self.error_count += 1
                return None
            
            result["conversations"] = conversations
            self.processed_count += 1
            return result
            
        except Exception as e:
            self.log_progress(f"Error processing item: {str(e)}", "error")
            self.error_count += 1
            return None
    
    def process_single_config(self, config_name: str) -> List[Dict[str, Any]]:
        """
        Process a single dataset configuration
        
        Args:
            config_name: Name of the configuration to process
            
        Returns:
            List of processed items
        """
        self.log_progress(f"Processing configuration: {config_name}")
        
        # Check if already processed
        if config_name in self.processed_configs:
            self.log_progress(f"Skipping configuration '{config_name}' as it has already been processed.")
            return []
        
        # Create image folder for this configuration
        image_folder = os.path.join(self.config.output_path, config_name)
        if os.path.exists(image_folder) and self.config.skip_existing:
            self.log_progress(f"Skipping configuration '{config_name}' as its folder already exists.")
            self.processed_configs.add(config_name)
            return []
        
        safe_makedirs(image_folder)
        
        try:
            # Load dataset for this configuration
            dataset = load_dataset_safe(
                self.config.input_path, 
                config=config_name, 
                split="train", 
                streaming=True
            )
            
            if dataset is None:
                self.log_progress(f"Failed to load dataset for config {config_name}", "error")
                return []
            
            # Process each item in the configuration
            converted_data = []
            config_processed = 0
            config_errors = 0
            
            for data in tqdm(dataset, desc=f"Processing {config_name}"):
                processed_item = self.process_config_item(data, config_name, image_folder)
                if processed_item:
                    converted_data.append(processed_item)
                    config_processed += 1
                else:
                    config_errors += 1
                
                # Periodic progress update
                if (config_processed + config_errors) % 1000 == 0:
                    self.log_progress(f"Config {config_name}: processed {config_processed}, errors {config_errors}")
            
            # Save configuration-specific JSON
            config_json_path = os.path.join(self.config.output_path, f"{config_name}.json")
            if save_json_data(converted_data, config_json_path):
                self.log_progress(f"Saved {len(converted_data)} items for config {config_name}")
            
            # Mark configuration as processed
            self.processed_configs.add(config_name)
            self._save_progress()
            
            self.log_progress(f"Finished processing configuration: {config_name}")
            return converted_data
            
        except Exception as e:
            self.log_progress(f"Error processing configuration {config_name}: {str(e)}", "error")
            return []
    
    def process_dataset(self) -> bool:
        """
        Process the entire dataset across all configurations
        
        Returns:
            True if successful, False otherwise
        """
        if not self.validate_config():
            return False
        
        # Get available configurations
        configs = self.get_dataset_configs()
        if not configs:
            self.log_progress("No configurations found", "error")
            return False
        
        # Process each configuration
        try:
            for config_name in configs:
                config_data = self.process_single_config(config_name)
                self.final_data.extend(config_data)
            
            # Save final combined data if requested
            if self.config.create_combined_file and self.final_data:
                dataset_name = os.path.basename(self.config.input_path).replace('/', '_')
                final_json_path = os.path.join(self.config.output_path, f"{dataset_name}.json")
                
                if save_json_data(self.final_data, final_json_path):
                    self.log_progress(f"Final combined data saved to {final_json_path}")
                else:
                    self.log_progress("Failed to save final combined data", "error")
                    return False
            
            # Log final statistics
            stats = self.get_stats()
            self.log_progress(f"Processing complete: {stats}")
            self.log_progress(f"Total items in final dataset: {len(self.final_data)}")
            
            return True
            
        except Exception as e:
            self.log_progress(f"Error during dataset processing: {str(e)}", "error")
            return False


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = setup_argparse("LLaVA-OneVision")
    parser.add_argument('--image-format', type=str, default='JPEG',
                       choices=['JPEG', 'PNG'], 
                       help='Output format for images')
    parser.add_argument('--no-combined-file', action='store_true',
                       help='Skip creating combined JSON file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Setup configuration
    config_manager = ConfigManager(args.config_file)
    
    # Get configuration with command line overrides
    config_overrides = {}
    if args.input_path:
        config_overrides['input_path'] = args.input_path
    if args.output_path:
        config_overrides['output_path'] = args.output_path
    if args.batch_size:
        config_overrides['batch_size'] = args.batch_size
    if args.max_workers:
        config_overrides['max_workers'] = args.max_workers
    if hasattr(args, 'image_format'):
        config_overrides['image_format'] = args.image_format
    if hasattr(args, 'no_combined_file'):
        config_overrides['create_combined_file'] = not args.no_combined_file
    if args.skip_existing:
        config_overrides['skip_existing'] = args.skip_existing
    
    config = config_manager.get_dataset_config("llava_onevision", **config_overrides)
    
    # Create and run parser
    parser = LLaVAOneVisionParser(config)
    success = parser.process_dataset()
    
    if success:
        print("LLaVA OneVision dataset processing completed successfully!")
        exit(0)
    else:
        print("LLaVA OneVision dataset processing failed!")
        exit(1)


if __name__ == "__main__":
    main() 