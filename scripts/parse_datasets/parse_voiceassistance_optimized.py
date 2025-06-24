#!/usr/bin/env python3
"""
Optimized Voice Assistant Dataset Parser

This script parses the VoiceAssistant-400K dataset and converts it to a format
suitable for training with LLaMA-Factory.

Features:
- Configurable input/output paths
- Comprehensive error handling and logging
- Progress tracking
- Batch processing for memory efficiency
- Resume capability for interrupted runs
"""

import os
import json
import soundfile
from tqdm import tqdm
from typing import Dict, Any, List, Optional

# Import common utilities
try:
    from ..common import (
        ConfigManager, VoiceAssistanceConfig, setup_argparse, 
        setup_logging, DatasetProcessor, load_dataset_safe,
        save_json_data, safe_makedirs
    )
except ImportError:
    # Fallback for when running without common module
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from common import (
        ConfigManager, VoiceAssistanceConfig, setup_argparse,
        setup_logging, DatasetProcessor, load_dataset_safe,
        save_json_data, safe_makedirs
    )


class VoiceAssistantParser(DatasetProcessor):
    """Voice Assistant dataset parser with enhanced functionality"""
    
    def __init__(self, config: VoiceAssistanceConfig):
        super().__init__(config)
        self.config = config
        self.audio_save_path = os.path.join(config.output_path, config.save_audio_dir)
        self.processed_ids = set()
        self._load_existing_progress()
    
    def _load_existing_progress(self):
        """Load existing progress to support resume functionality"""
        progress_file = os.path.join(self.config.output_path, "progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    self.processed_ids = set(progress.get('processed_ids', []))
                    self.log_progress(f"Loaded {len(self.processed_ids)} processed items from previous run")
            except Exception as e:
                self.log_progress(f"Could not load progress file: {str(e)}", "warning")
    
    def _save_progress(self):
        """Save current progress"""
        progress_file = os.path.join(self.config.output_path, "progress.json")
        try:
            progress = {
                'processed_ids': list(self.processed_ids),
                'total_processed': self.processed_count,
                'total_errors': self.error_count
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.log_progress(f"Could not save progress: {str(e)}", "warning")
    
    def parse_voice_item(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a single voice assistant item
        
        Args:
            data: Raw data item from dataset
            
        Returns:
            Parsed item in target format or None if failed
        """
        try:
            item_id = data.get("index")
            if not item_id:
                self.log_progress("Item missing index field", "warning")
                return None
            
            # Skip if already processed (for resume functionality)
            if item_id in self.processed_ids:
                return None
            
            audio = data.get("question_audio")
            if not audio:
                self.log_progress(f"Item {item_id} missing audio data", "warning")
                self.error_count += 1
                return None
            
            # Extract audio information
            audio_path = audio.get("path")
            audio_np = audio.get("array")
            sampling_rate = audio.get("sampling_rate")
            
            if not all([audio_path, audio_np is not None, sampling_rate]):
                self.log_progress(f"Item {item_id} has incomplete audio data", "warning")
                self.error_count += 1
                return None
            
            # Save audio file
            save_path = os.path.join(self.audio_save_path, audio_path)
            try:
                safe_makedirs(os.path.dirname(save_path))
                soundfile.write(save_path, audio_np, sampling_rate)
            except Exception as e:
                self.log_progress(f"Failed to save audio for item {item_id}: {str(e)}", "error")
                self.error_count += 1
                return None
            
            # Generate relative path for the dataset
            rel_path = os.path.join("VoiceAssistant-400K-Parsed", "audio", audio_path)
            
            # Generate text prompt and messages
            question = data.get("question", "")
            answer = data.get("answer", "")
            
            if not question or not answer:
                self.log_progress(f"Item {item_id} missing question or answer", "warning")
                self.error_count += 1
                return None
            
            prompt = f"<audio>{question}"
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ]
            
            # Mark as processed
            self.processed_ids.add(item_id)
            self.processed_count += 1
            
            return {
                "id": item_id,
                "messages": messages,
                "audios": [rel_path]
            }
            
        except Exception as e:
            self.log_progress(f"Error parsing item: {str(e)}", "error")
            self.error_count += 1
            return None
    
    def process_dataset(self) -> bool:
        """
        Process the entire dataset
        
        Returns:
            True if successful, False otherwise
        """
        if not self.validate_config():
            return False
        
        # Create output directories
        safe_makedirs(self.audio_save_path)
        
        # Load dataset
        self.log_progress(f"Loading dataset from {self.config.input_path}")
        dataset = load_dataset_safe(self.config.input_path, split="train", streaming=True)
        
        if dataset is None:
            self.log_progress("Failed to load dataset", "error")
            return False
        
        # Process dataset
        converted_data = []
        
        try:
            self.log_progress("Starting dataset processing...")
            
            for data in tqdm(dataset, desc="Processing voice assistant data"):
                parsed_item = self.parse_voice_item(data)
                if parsed_item:
                    converted_data.append(parsed_item)
                
                # Save progress periodically
                if len(converted_data) % 1000 == 0:
                    self._save_progress()
                    self.log_progress(f"Processed {len(converted_data)} items")
            
            # Save final results
            output_file = os.path.join(self.config.output_path, "voiceassistant.json")
            if save_json_data(converted_data, output_file):
                self.log_progress(f"Successfully saved {len(converted_data)} items to {output_file}")
                
                # Save final progress
                self._save_progress()
                
                # Log statistics
                stats = self.get_stats()
                self.log_progress(f"Processing complete: {stats}")
                
                return True
            else:
                self.log_progress("Failed to save final results", "error")
                return False
                
        except Exception as e:
            self.log_progress(f"Error during dataset processing: {str(e)}", "error")
            return False


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = setup_argparse("VoiceAssistant")
    parser.add_argument('--save-audio-dir', type=str, default='audio',
                       help='Directory name for saving audio files')
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
    if hasattr(args, 'save_audio_dir'):
        config_overrides['save_audio_dir'] = args.save_audio_dir
    
    config = config_manager.get_dataset_config("voiceassistant", **config_overrides)
    
    # Create and run parser
    parser = VoiceAssistantParser(config)
    success = parser.process_dataset()
    
    if success:
        print("Voice assistant dataset processing completed successfully!")
        exit(0)
    else:
        print("Voice assistant dataset processing failed!")
        exit(1)


if __name__ == "__main__":
    main() 