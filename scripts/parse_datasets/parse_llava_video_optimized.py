#!/usr/bin/env python3
"""
Optimized LLaVA Video Dataset Parser

This script processes the LLaVA Video 178K dataset by extracting tar.gz files
and copying other files while preserving the directory structure.

Features:
- Configurable input/output paths
- Comprehensive error handling and logging
- Progress tracking for file operations
- Resume capability for interrupted runs
- Validation of tar files before extraction
- Detailed statistics reporting
"""

import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import common utilities
try:
    from ..common import (
        ConfigManager, LLaVAConfig, setup_argparse, 
        setup_logging, DatasetProcessor, extract_tar_safe,
        copy_file_safe, safe_makedirs
    )
except ImportError:
    # Fallback for when running without common module
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from common import (
        ConfigManager, LLaVAConfig, setup_argparse,
        setup_logging, DatasetProcessor, extract_tar_safe,
        copy_file_safe, safe_makedirs
    )


class LLaVAVideoParser(DatasetProcessor):
    """LLaVA Video dataset parser with enhanced functionality"""
    
    def __init__(self, config: LLaVAConfig):
        super().__init__(config)
        self.config = config
        self.stats = {
            'tar_files': 0,
            'extracted_tar_files': 0,
            'copied_files': 0,
            'skipped_files': 0,
            'failed_extractions': 0,
            'failed_copies': 0
        }
        self.processed_files = set()
        self._load_existing_progress()
    
    def _load_existing_progress(self):
        """Load existing progress to support resume functionality"""
        progress_file = os.path.join(self.config.output_path, "progress.json")
        if os.path.exists(progress_file):
            try:
                import json
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    self.processed_files = set(progress.get('processed_files', []))
                    self.stats.update(progress.get('stats', {}))
                    self.log_progress(f"Loaded {len(self.processed_files)} processed files from previous run")
            except Exception as e:
                self.log_progress(f"Could not load progress file: {str(e)}", "warning")
    
    def _save_progress(self):
        """Save current progress"""
        progress_file = os.path.join(self.config.output_path, "progress.json")
        try:
            import json
            progress = {
                'processed_files': list(self.processed_files),
                'stats': self.stats,
                'total_processed': self.processed_count,
                'total_errors': self.error_count
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.log_progress(f"Could not save progress: {str(e)}", "warning")
    
    def get_file_list(self) -> List[Tuple[str, str, str]]:
        """
        Get list of all files to process
        
        Returns:
            List of tuples (source_file, relative_path, file_type)
        """
        files_to_process = []
        
        try:
            for root, dirs, files in os.walk(self.config.input_path):
                rel_dir = os.path.relpath(root, self.config.input_path)
                
                for file in files:
                    src_file = os.path.join(root, file)
                    rel_path = os.path.join(rel_dir, file)
                    
                    # Determine file type
                    if file.endswith('.tar.gz'):
                        file_type = 'tar'
                        self.stats['tar_files'] += 1
                    else:
                        file_type = 'copy'
                    
                    files_to_process.append((src_file, rel_path, file_type))
            
            self.log_progress(f"Found {len(files_to_process)} files to process")
            self.log_progress(f"Tar files: {self.stats['tar_files']}, Other files: {len(files_to_process) - self.stats['tar_files']}")
            
            return files_to_process
            
        except Exception as e:
            self.log_progress(f"Error scanning input directory: {str(e)}", "error")
            return []
    
    def extract_tar_file(self, src_file: str, dst_dir: str, rel_path: str) -> bool:
        """
        Extract a tar.gz file to destination directory
        
        Args:
            src_file: Source tar.gz file path
            dst_dir: Destination directory
            rel_path: Relative path for progress tracking
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Skip if already processed
            if rel_path in self.processed_files:
                self.stats['skipped_files'] += 1
                return True
            
            # Validate tar file
            if not tarfile.is_tarfile(src_file):
                self.log_progress(f"Warning: {src_file} is not a valid tar.gz file, will copy instead", "warning")
                # Copy as regular file
                dst_file = os.path.join(dst_dir, os.path.basename(src_file))
                if copy_file_safe(src_file, dst_file):
                    self.stats['copied_files'] += 1
                    self.processed_files.add(rel_path)
                    return True
                else:
                    self.stats['failed_copies'] += 1
                    return False
            
            # Extract tar file
            if extract_tar_safe(src_file, dst_dir):
                self.log_progress(f"Extracted {src_file} to {dst_dir}")
                self.stats['extracted_tar_files'] += 1
                self.processed_files.add(rel_path)
                self.processed_count += 1
                return True
            else:
                self.stats['failed_extractions'] += 1
                self.error_count += 1
                return False
                
        except Exception as e:
            self.log_progress(f"Error extracting {src_file}: {str(e)}", "error")
            self.stats['failed_extractions'] += 1
            self.error_count += 1
            return False
    
    def copy_regular_file(self, src_file: str, dst_file: str, rel_path: str) -> bool:
        """
        Copy a regular file to destination
        
        Args:
            src_file: Source file path
            dst_file: Destination file path
            rel_path: Relative path for progress tracking
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Skip if already processed
            if rel_path in self.processed_files:
                self.stats['skipped_files'] += 1
                return True
            
            # Copy file
            if copy_file_safe(src_file, dst_file):
                self.log_progress(f"Copied {src_file} to {dst_file}")
                self.stats['copied_files'] += 1
                self.processed_files.add(rel_path)
                self.processed_count += 1
                return True
            else:
                self.stats['failed_copies'] += 1
                self.error_count += 1
                return False
                
        except Exception as e:
            self.log_progress(f"Error copying {src_file}: {str(e)}", "error")
            self.stats['failed_copies'] += 1
            self.error_count += 1
            return False
    
    def process_single_file(self, file_info: Tuple[str, str, str]) -> bool:
        """
        Process a single file (extract or copy)
        
        Args:
            file_info: Tuple of (source_file, relative_path, file_type)
            
        Returns:
            True if successful, False otherwise
        """
        src_file, rel_path, file_type = file_info
        dst_dir = os.path.join(self.config.output_path, os.path.dirname(rel_path))
        
        # Create destination directory
        safe_makedirs(dst_dir)
        
        if file_type == 'tar':
            return self.extract_tar_file(src_file, dst_dir, rel_path)
        else:
            dst_file = os.path.join(dst_dir, os.path.basename(src_file))
            return self.copy_regular_file(src_file, dst_file, rel_path)
    
    def process_dataset(self) -> bool:
        """
        Process the entire dataset
        
        Returns:
            True if successful, False otherwise
        """
        if not self.validate_config():
            return False
        
        # Get list of files to process
        files_to_process = self.get_file_list()
        if not files_to_process:
            self.log_progress("No files found to process", "warning")
            return False
        
        # Process files
        try:
            self.log_progress("Starting file processing...")
            
            if self.config.max_workers > 1:
                # Process files in parallel
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    future_to_file = {
                        executor.submit(self.process_single_file, file_info): file_info 
                        for file_info in files_to_process
                    }
                    
                    with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
                        for future in as_completed(future_to_file):
                            file_info = future_to_file[future]
                            try:
                                success = future.result()
                                if not success:
                                    self.log_progress(f"Failed to process {file_info[1]}", "warning")
                            except Exception as e:
                                self.log_progress(f"Exception processing {file_info[1]}: {str(e)}", "error")
                                self.error_count += 1
                            
                            pbar.update(1)
                            
                            # Save progress periodically
                            if pbar.n % 100 == 0:
                                self._save_progress()
            else:
                # Process files sequentially
                with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
                    for file_info in files_to_process:
                        success = self.process_single_file(file_info)
                        if not success:
                            self.log_progress(f"Failed to process {file_info[1]}", "warning")
                        
                        pbar.update(1)
                        
                        # Save progress periodically
                        if pbar.n % 100 == 0:
                            self._save_progress()
            
            # Save final progress
            self._save_progress()
            
            # Log final statistics
            self.log_progress("Processing complete!")
            self.log_progress(f"Statistics: {self.stats}")
            self.log_progress(f"Total processed: {self.processed_count}, Errors: {self.error_count}")
            
            return True
            
        except Exception as e:
            self.log_progress(f"Error during dataset processing: {str(e)}", "error")
            return False
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        total_files = sum([
            self.stats['extracted_tar_files'],
            self.stats['copied_files'],
            self.stats['skipped_files'],
            self.stats['failed_extractions'],
            self.stats['failed_copies']
        ])
        
        return {
            **self.stats,
            'total_files': total_files,
            'success_rate': (self.processed_count / max(1, total_files)) * 100,
            'processed_files_count': len(self.processed_files)
        }


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = setup_argparse("LLaVA-Video")
    parser.add_argument('--parallel', action='store_true',
                       help='Process files in parallel')
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
    if args.skip_existing:
        config_overrides['skip_existing'] = args.skip_existing
    
    config = config_manager.get_dataset_config("llava_video", **config_overrides)
    
    # Create and run parser
    parser = LLaVAVideoParser(config)
    success = parser.process_dataset()
    
    if success:
        stats = parser.get_detailed_stats()
        print("LLaVA Video dataset processing completed successfully!")
        print(f"Detailed statistics: {stats}")
        exit(0)
    else:
        print("LLaVA Video dataset processing failed!")
        exit(1)


if __name__ == "__main__":
    main() 