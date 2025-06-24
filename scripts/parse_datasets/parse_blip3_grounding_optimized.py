#!/usr/bin/env python3
"""
Optimized BLIP3 Grounding Dataset Parser

This script processes the BLIP3 grounding 50M dataset by filtering out unnecessary columns
and converting it to a more manageable format.

Features:
- Configurable input/output paths
- Support for both streaming and parquet loading
- Comprehensive error handling and logging
- Memory-efficient processing
- Progress tracking
"""

import os
import pandas as pd
from datasets import load_dataset
from typing import Dict, Any, List, Optional, Union

# Import common utilities
try:
    from ..common import (
        ConfigManager, DatasetConfig, setup_argparse, 
        setup_logging, DatasetProcessor, load_dataset_safe,
        safe_makedirs, chunk_list
    )
except ImportError:
    # Fallback for when running without common module
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from common import (
        ConfigManager, DatasetConfig, setup_argparse,
        setup_logging, DatasetProcessor, load_dataset_safe,
        safe_makedirs, chunk_list
    )


class BLIP3GroundingParser(DatasetProcessor):
    """BLIP3 Grounding dataset parser with enhanced functionality"""
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.columns_to_remove = ['cogvlm_caption', 'captions']
        self.output_formats = ['parquet', 'json']  # Supported output formats
    
    def examine_dataset_structure(self, dataset) -> Dict[str, Any]:
        """
        Examine and log dataset structure
        
        Args:
            dataset: HuggingFace dataset object
            
        Returns:
            Dictionary with dataset information
        """
        try:
            self.log_progress("Examining dataset structure...")
            
            # Get dataset info
            info = {
                'features': list(dataset['train'].features.keys()) if hasattr(dataset, '__getitem__') else None,
                'num_shards': getattr(dataset.get('train', dataset), 'num_shards', 'unknown'),
                'sample_data': None
            }
            
            # Get sample data
            train_data = dataset['train'] if hasattr(dataset, '__getitem__') else dataset
            try:
                train_iter = iter(train_data)
                first_item = next(train_iter)
                
                # Log sample item (truncated for readability)
                sample_info = {}
                for key, value in first_item.items():
                    if key in ['cogvlm_caption', 'captions']:
                        sample_info[key] = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    else:
                        sample_info[key] = value
                
                info['sample_data'] = sample_info
                self.log_progress(f"Dataset features: {info['features']}")
                self.log_progress(f"Sample item: {sample_info}")
                
            except Exception as e:
                self.log_progress(f"Could not get sample data: {str(e)}", "warning")
            
            return info
            
        except Exception as e:
            self.log_progress(f"Error examining dataset: {str(e)}", "error")
            return {}
    
    def filter_dataset_columns(self, dataset) -> Optional[object]:
        """
        Remove unnecessary columns from dataset
        
        Args:
            dataset: Input dataset
            
        Returns:
            Filtered dataset or None if failed
        """
        try:
            self.log_progress(f"Removing columns: {self.columns_to_remove}")
            
            # Check if columns exist
            train_data = dataset['train'] if hasattr(dataset, '__getitem__') else dataset
            existing_features = list(train_data.features.keys())
            
            columns_to_remove = [col for col in self.columns_to_remove if col in existing_features]
            if not columns_to_remove:
                self.log_progress("No columns to remove, returning original dataset")
                return dataset
            
            # Remove columns
            filtered_dataset = dataset.map(
                lambda example: {k: v for k, v in example.items() if k not in columns_to_remove},
                remove_columns=columns_to_remove
            )
            
            self.log_progress(f"Successfully removed columns: {columns_to_remove}")
            return filtered_dataset
            
        except Exception as e:
            self.log_progress(f"Error filtering dataset columns: {str(e)}", "error")
            return None
    
    def process_streaming_dataset(self) -> bool:
        """
        Process dataset using streaming approach
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.log_progress(f"Loading streaming dataset from {self.config.input_path}")
            
            # Load dataset
            dataset = load_dataset_safe(self.config.input_path, streaming=True)
            if dataset is None:
                return False
            
            # Examine structure
            dataset_info = self.examine_dataset_structure(dataset)
            
            # Filter columns
            filtered_dataset = self.filter_dataset_columns(dataset)
            if filtered_dataset is None:
                return False
            
            # Log filtered structure
            self.log_progress("Dataset after filtering:")
            train_data = filtered_dataset['train'] if hasattr(filtered_dataset, '__getitem__') else filtered_dataset
            
            try:
                filtered_iter = iter(train_data)
                filtered_first = next(filtered_iter)
                self.log_progress(f"Filtered features: {list(filtered_first.keys())}")
                self.log_progress(f"Sample filtered item: {filtered_first}")
            except Exception as e:
                self.log_progress(f"Could not get filtered sample: {str(e)}", "warning")
            
            # Save filtered dataset information
            info_file = os.path.join(self.config.output_path, "dataset_info.json")
            try:
                import json
                with open(info_file, 'w') as f:
                    json.dump(dataset_info, f, indent=2, default=str)
                self.log_progress(f"Saved dataset info to {info_file}")
            except Exception as e:
                self.log_progress(f"Could not save dataset info: {str(e)}", "warning")
            
            self.processed_count += 1
            return True
            
        except Exception as e:
            self.log_progress(f"Error processing streaming dataset: {str(e)}", "error")
            self.error_count += 1
            return False
    
    def process_parquet_file(self, parquet_path: str) -> Optional[pd.DataFrame]:
        """
        Process a single parquet file
        
        Args:
            parquet_path: Path to parquet file
            
        Returns:
            Processed DataFrame or None if failed
        """
        try:
            self.log_progress(f"Loading parquet file: {parquet_path}")
            
            # Load parquet file
            df = pd.read_parquet(parquet_path)
            self.log_progress(f"Loaded {len(df)} rows from {parquet_path}")
            
            # Remove unnecessary columns
            columns_to_remove = [col for col in self.columns_to_remove if col in df.columns]
            if columns_to_remove:
                df = df.drop(columns=columns_to_remove)
                self.log_progress(f"Removed columns: {columns_to_remove}")
            
            # Log remaining columns and sample
            self.log_progress(f"Remaining columns: {list(df.columns)}")
            self.log_progress(f"Sample row: {df.iloc[0].to_dict()}")
            
            self.processed_count += len(df)
            return df
            
        except Exception as e:
            self.log_progress(f"Error processing parquet file {parquet_path}: {str(e)}", "error")
            self.error_count += 1
            return None
    
    def process_dataset(self) -> bool:
        """
        Process the dataset based on input type
        
        Returns:
            True if successful, False otherwise
        """
        if not self.validate_config():
            return False
        
        # Determine input type
        if os.path.isfile(self.config.input_path) and self.config.input_path.endswith('.parquet'):
            # Single parquet file
            self.log_progress("Processing single parquet file")
            df = self.process_parquet_file(self.config.input_path)
            if df is not None:
                # Save processed data
                output_path = os.path.join(self.config.output_path, "processed_data.parquet")
                safe_makedirs(os.path.dirname(output_path))
                df.to_parquet(output_path, index=False)
                self.log_progress(f"Saved processed data to {output_path}")
                return True
            return False
            
        elif os.path.isdir(self.config.input_path):
            # Directory with potential parquet files or HuggingFace dataset
            parquet_files = [f for f in os.listdir(self.config.input_path) if f.endswith('.parquet')]
            
            if parquet_files:
                # Process multiple parquet files
                self.log_progress(f"Found {len(parquet_files)} parquet files")
                all_dfs = []
                
                for parquet_file in parquet_files:
                    parquet_path = os.path.join(self.config.input_path, parquet_file)
                    df = self.process_parquet_file(parquet_path)
                    if df is not None:
                        all_dfs.append(df)
                
                if all_dfs:
                    # Combine all DataFrames
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    output_path = os.path.join(self.config.output_path, "processed_data.parquet")
                    safe_makedirs(os.path.dirname(output_path))
                    combined_df.to_parquet(output_path, index=False)
                    self.log_progress(f"Combined and saved {len(combined_df)} rows to {output_path}")
                    return True
                    
            else:
                # Try as HuggingFace dataset
                return self.process_streaming_dataset()
        
        else:
            # Try as HuggingFace dataset path
            return self.process_streaming_dataset()
        
        return False


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = setup_argparse("BLIP3-Grounding")
    parser.add_argument('--columns-to-remove', nargs='+', 
                       default=['cogvlm_caption', 'captions'],
                       help='Columns to remove from dataset')
    parser.add_argument('--output-format', choices=['parquet', 'json'], 
                       default='parquet',
                       help='Output format for processed data')
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
    
    config = config_manager.get_dataset_config("blip3", **config_overrides)
    
    # Create and run parser
    parser = BLIP3GroundingParser(config)
    
    # Override default columns if specified
    if hasattr(args, 'columns_to_remove') and args.columns_to_remove:
        parser.columns_to_remove = args.columns_to_remove
    
    success = parser.process_dataset()
    
    if success:
        stats = parser.get_stats()
        print(f"BLIP3 grounding dataset processing completed successfully!")
        print(f"Statistics: {stats}")
        exit(0)
    else:
        print("BLIP3 grounding dataset processing failed!")
        exit(1)


if __name__ == "__main__":
    main() 