import os
import json
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
from tqdm import tqdm
import pandas as pd
from datasets import Dataset


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    
    return logging.getLogger(__name__)


def safe_makedirs(path: str, exist_ok: bool = True) -> bool:
    """Safely create directories with error handling"""
    try:
        os.makedirs(path, exist_ok=exist_ok)
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {str(e)}")
        return False


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logging.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
                    if delay > 0:
                        import time
                        time.sleep(delay)
            return None
        return wrapper
    return decorator


def validate_paths(*paths: str) -> bool:
    """Validate that all input paths exist"""
    for path in paths:
        if not os.path.exists(path):
            logging.error(f"Path does not exist: {path}")
            return False
    return True


def save_json_data(data: List[Dict[str, Any]], output_path: str, indent: int = 2) -> bool:
    """Safely save JSON data with error handling"""
    try:
        safe_makedirs(os.path.dirname(output_path))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logging.info(f"Successfully saved {len(data)} items to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON data to {output_path}: {str(e)}")
        return False


def save_jsonl_data(data: List[Dict[str, Any]], output_path: str) -> bool:
    """Save data in JSONL format"""
    try:
        safe_makedirs(os.path.dirname(output_path))
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"Successfully saved {len(data)} items to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save JSONL data to {output_path}: {str(e)}")
        return False


def load_dataset_safe(dataset_path: str, config: Optional[str] = None, 
                     split: str = "train", streaming: bool = False):
    """Safely load dataset with error handling"""
    try:
        from datasets import load_dataset
        if config:
            return load_dataset(dataset_path, config, split=split, streaming=streaming)
        else:
            return load_dataset(dataset_path, split=split, streaming=streaming)
    except Exception as e:
        logging.error(f"Failed to load dataset from {dataset_path}: {str(e)}")
        return None


def extract_tar_safe(tar_path: str, extract_to: str) -> bool:
    """Safely extract tar files with validation"""
    try:
        if not tarfile.is_tarfile(tar_path):
            logging.warning(f"File is not a valid tar file: {tar_path}")
            return False
            
        safe_makedirs(extract_to)
        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(path=extract_to)
        logging.info(f"Successfully extracted {tar_path} to {extract_to}")
        return True
    except Exception as e:
        logging.error(f"Failed to extract {tar_path}: {str(e)}")
        return False


def copy_file_safe(src: str, dst: str) -> bool:
    """Safely copy files with error handling"""
    try:
        safe_makedirs(os.path.dirname(dst))
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logging.error(f"Failed to copy {src} to {dst}: {str(e)}")
        return False


def process_in_batches(data, batch_size: int, process_func: Callable, 
                      desc: str = "Processing") -> List[Any]:
    """Process data in batches with progress bar"""
    results = []
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    with tqdm(total=total_batches, desc=desc) as pbar:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            try:
                batch_result = process_func(batch)
                if batch_result:
                    results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
            except Exception as e:
                logging.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            pbar.update(1)
    
    return results


def get_file_stats(file_path: str) -> Dict[str, Any]:
    """Get file statistics"""
    try:
        stat = os.stat(file_path)
        return {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "exists": True
        }
    except:
        return {"exists": False}


class DatasetProcessor:
    """Base class for dataset processors with common functionality"""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.processed_count = 0
        self.error_count = 0
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        if not validate_paths(self.config.input_path):
            return False
        safe_makedirs(self.config.output_path)
        return True
    
    def log_progress(self, message: str, level: str = "info"):
        """Log progress messages"""
        getattr(self.logger, level)(f"[{self.__class__.__name__}] {message}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        return {
            "processed": self.processed_count,
            "errors": self.error_count,
            "success_rate": (self.processed_count / max(1, self.processed_count + self.error_count)) * 100
        }


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)] 