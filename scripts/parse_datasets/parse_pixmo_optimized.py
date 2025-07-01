#!/usr/bin/env python3
"""
Optimized Pixmo Points Dataset Parser

This script processes the Pixmo Points dataset by downloading images and converting
point annotations to LLaMA-Factory format with async download capabilities.

Features:
- Configurable async download parameters
- Hash verification for downloaded images  
- Resume capability for interrupted runs
- Comprehensive error handling and logging
- Progress tracking for downloads and processing
- Multiple output formats (XML, JSON, coordinate lists)
- Graceful shutdown handling
"""

import asyncio
import aiohttp
import aiofiles
import json
import time
import signal
import sys
import cv2
import numpy as np
from datasets import load_dataset, Dataset
from hashlib import sha256
from typing import Dict, Any, List, Optional, Tuple
from tqdm.asyncio import tqdm
from PIL import Image
from qwen_vl_utils import smart_resize

# Import common utilities
try:
    from ..common import (
        ConfigManager, PixmoConfig, setup_argparse, 
        setup_logging, DatasetProcessor, save_json_data,
        safe_makedirs, chunk_list, save_jsonl_data
    )
except ImportError:
    # Fallback for when running without common module
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from common import (
        ConfigManager, PixmoConfig, setup_argparse,
        setup_logging, DatasetProcessor, save_json_data,
        safe_makedirs, chunk_list, save_jsonl_data
    )


# Prompt templates for different annotation formats
GENERAL_PROMPTS = { 
    "pointing": [
        "Point to {label}\nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{label}\"",
        "Point to any {label} in the image",
        "Point to any {label} in the image.",
        "Point: Where are the {label}",
        "Show me where the {label} are",
        "Can you show me where the {label} are?",
        "Show me where the {label} are",
        "Show me where a {label} is",
        "Show me where a {label} is.",
        "If there are any {label} in the image? Show me where they are.",
        "Where are the {label}?",
        "Generate a list of points showing where the {label} are.",
        "Find the \"{label}\".",
        "Find a \"{label}\".",
        "Locate all {label}.",
        "Locate an {label}.",
        "Locate a {label}.",
        "Locate every {label}.",
        "Locate {label}.",
        "Locate the {label}.",
        "Object: {label}\nInstruction: Point to the object.",
        "find {label}",
        "find {label}.",
        "Point to every {label}",
        "find any {label} in the picture",
        "Find the {label}",
        "Find any {label}",
        "Point to a {label}",
        "Point to an {label}",
        "Look for {label} in the image and show me where they are.",
        "Help me find an object in the image by pointing to them.\nObject: {label}.",
        "I am looking for {label}, where can they be found in the image?",
        "Can you see any {label} in the image? Point to them.",
        "Point out each {label} in the image.",
        "Point out every {label} in the image.",
        "Point to the {label} in the image.",
        "Locate each {label} in the image.",
        "Can you point out all {label} in this image?",
        "Please find {label} and show me where they are.",
        "If there are any {label} present, indicate their positions.",
        "If there is a {label} present, indicate its positions.",
        "show me all visible {label}",
    ],
    "point_count": [
        "How many {label} are there?",
        "How many {label}?",
        "How many {label}.",
        "how many {label}.",
        "how many {label}?",
        "How many {label} are there in the image?",
        "Tell me how many {label} there are",
        "Tell me how many {label} there are and point to them.",
        "how many {label}",
        "Tell me where each {label} is.",
        "Tell me how many {label} are in the image",
        "count {label}",
        "count every {label}",
        "count each {label}",
        "count {label}.",
        "Count the {label}.",
        "How many {label} do you see?",
        "How many {label} are visible?",
        "Count all the {label}",
        "how mmny {label}?",
        "Count every {label} in the picture.",
        "Count all the {label}",
        "Count each {label}",
        "Point to and count the {label} in the picture.",
        "Point and count {label}",
        "Point to every {label}",
        "Locate the {label} and count them",
        "Locate every {label} and count them",
        "Find all the {label}. How many are there?",
        "Find each {label}. How many are there?",
        "Point at {label} and then tell me the count.",
        "What is the total number of {label} in the image?",
        "In all the picture, how many {label} are there?",
        "Point at the {label} and then count them.",
        "Point to all the visible {label} output the total count.",
        "Point to all the {label} visible and output the total count. \nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{label}\" and output the total count.",
        "Show me where the {label} are and output the total count.",
        "Where are the {label}? How many are there?",
        "Generate list of points showing where the {label} are and output the total count.",
        "Object: {label}\nInstruction: Point to the object and output the total count.",
        "find any {label} in the picture and output the total count.",
        "Can you see any {label} in the image? Point to them and output the total count.",
        "Can you point out all {label} in this image? How many are there?",
        "If there are any {label} present, indicate their positions and output the total count.",
        "How many {label} are there in the image? Point to them and output the total count.",
        "How many {label} are there in the image?",
        "Give me the count of {label} in the image.",
        "How many {label} are visible in the image?",
        "How many {label} are there?",
        "In the image, how many {label} are there?",
        "Can you count the number of {label} in the image?",
        "Can you count every {label} in the picture?",
        "Can you see any {label} in the image? How many are there?",
        "Are there any {label} in the image? How many are there?",
        "If you see any {label} in the image, give me the count. Otherwise, say 'This isn't in the image.'",
        "Object: {label}\nInstruction: How many are there?",
    ],
    "count_then_point": [
        "Count the {label} in the image, then point to them.",
        "How many {label} are there? Point to them.",
        "Count every {label} in the picture, then point to them.",
        "Locate the {label} and count them, then point to them.",
        "Find all the {label}. How many are there? Point to them.",
        "Find each {label}. How many are there? Point to them.",
        "Point to and count the {label} in the picture.",
    ],
    "pointing_xml": [
        "Point to {label} using XML format.\nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{label}\" in XML format",
        "Locate {label} and output coordinates in XML format",
        "Find {label} and return positions as XML points",
        "Show me where the {label} are using XML markup",
        "Generate XML point tags for all {label} in the image",
        "Use XML format to indicate the location of {label}",
        "Output XML coordinates for each {label} you find",
        "Identify {label} positions using XML point notation",
        "Mark the locations of {label} with XML point tags",
        "Find {label} and format output as XML points with coordinates",
        "Locate every {label} and represent as XML point elements",
        "Point to {label} and format response using XML point syntax",
        "Use <point> or <points> XML tags to mark {label} locations",
        "Return {label} coordinates in structured XML format",
    ],
    "pointing_json": [
        "Point to {label} using JSON format.\nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{label}\" in JSON format",
        "Locate {label} and output coordinates in JSON format",
        "Find {label} and return positions as JSON objects",
        "Show me where the {label} are using JSON notation",
        "Generate JSON objects for all {label} coordinates in the image",
        "Use JSON format to indicate the location of {label}",
        "Output JSON coordinates for each {label} you find",
        "Identify {label} positions using JSON object notation",
        "Mark the locations of {label} with JSON coordinate objects",
        "Find {label} and format output as JSON with x, y coordinates",
        "Locate every {label} and represent as JSON point objects",
        "Point to {label} and format response using JSON syntax",
        "Use JSON objects with x, y properties to mark {label} locations",
        "Return {label} coordinates in structured JSON format",
    ],
    "pointing_list": [
        "Point to {label} using coordinate list format.\nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{label}\" as coordinate pairs",
        "Locate {label} and output coordinates as comma-separated values",
        "Find {label} and return positions as x,y coordinate pairs",
        "Show me where the {label} are using coordinate list format",
        "Generate coordinate pairs for all {label} in the image",
        "Use coordinate list format to indicate the location of {label}",
        "Output coordinate pairs for each {label} you find",
        "Identify {label} positions using x,y coordinate format",
        "Mark the locations of {label} with coordinate pairs",
        "Find {label} and format output as coordinate lists",
        "Locate every {label} and represent as x,y pairs",
        "Point to {label} and format response as coordinate pairs",
        "Use x,y coordinate format to mark {label} locations",
        "Return {label} coordinates as comma-separated coordinate pairs",
    ],
}


class DownloadStats:
    """Thread-safe statistics tracking for downloads"""
    
    def __init__(self):
        self.total = 0
        self.skipped_existing = 0
        self.download_success = 0
        self.download_failed = 0
        self.hash_mismatch = 0
        self.save_failed = 0
        self.start_time = time.time()
        self._lock = asyncio.Lock()
        self.hash_debug_count = 0
        
    async def increment(self, field: str):
        async with self._lock:
            setattr(self, field, getattr(self, field) + 1)
        
    def get_success_rate(self) -> float:
        return (self.download_success / max(1, self.total)) * 100
        
    def get_throughput(self) -> float:
        elapsed = time.time() - self.start_time
        return self.download_success / max(1, elapsed)
        
    def __str__(self) -> str:
        elapsed = time.time() - self.start_time
        return (f"Total: {self.total}, Success: {self.download_success}, "
                f"Failed: {self.download_failed}, Skipped: {self.skipped_existing}, "
                f"Hash mismatch: {self.hash_mismatch}, Save failed: {self.save_failed}, "
                f"Success rate: {self.get_success_rate():.2f}%, "
                f"Throughput: {self.get_throughput():.2f} images/sec, "
                f"Elapsed: {elapsed:.2f}s")


class GracefulExit:
    """Handle graceful shutdown on signals"""
    
    def __init__(self):
        self.exit_requested = False
        signal.signal(signal.SIGINT, self.request_exit)
        signal.signal(signal.SIGTERM, self.request_exit)
    
    def request_exit(self, signum, frame):
        print("Exit requested, finishing current downloads...")
        self.exit_requested = True


class PixmoParser(DatasetProcessor):
    """Pixmo Points dataset parser with async download capabilities"""
    
    def __init__(self, config: PixmoConfig):
        super().__init__(config)
        self.config = config
        self.stats = DownloadStats()
        self.graceful_exit = GracefulExit()
        self.calculated_hashes: Dict[int, str] = {}
        self.calculated_hashes_lock = asyncio.Lock()
        self._load_existing_progress()
    
    def _load_existing_progress(self):
        """Load existing progress to support resume functionality"""
        progress_file = os.path.join(self.config.output_path, "progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    self.calculated_hashes = {int(k): v for k, v in progress.get('calculated_hashes', {}).items()}
                    self.log_progress(f"Loaded {len(self.calculated_hashes)} calculated hashes from previous run")
            except Exception as e:
                self.log_progress(f"Could not load progress file: {str(e)}", "warning")
    
    def _save_progress(self):
        """Save current progress"""
        progress_file = os.path.join(self.config.output_path, "progress.json") 
        try:
            progress = {
                'calculated_hashes': {str(k): v for k, v in self.calculated_hashes.items()},
                'stats': {
                    'total': self.stats.total,
                    'download_success': self.stats.download_success,
                    'download_failed': self.stats.download_failed,
                    'hash_mismatch': self.stats.hash_mismatch,
                    'skipped_existing': self.stats.skipped_existing
                }
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.log_progress(f"Could not save progress: {str(e)}", "warning")
    
    async def save_debug_info(self, image_data: bytes, expected_hash: str, 
                             calculated_hash: str, image_url: str):
        """Save debugging information for hash mismatches"""
        if not self.config.save_failed_hash_samples:
            return
            
        debug_dir = os.path.join(self.config.output_path, "debug_hash_mismatch")
        safe_makedirs(debug_dir)
        
        # Save the actual downloaded image
        debug_file = os.path.join(debug_dir, f"mismatch_{self.stats.hash_debug_count}_{calculated_hash[:8]}.png")
        async with aiofiles.open(debug_file, 'wb') as f:
            await f.write(image_data)
        
        # Save metadata
        metadata_file = os.path.join(debug_dir, f"mismatch_{self.stats.hash_debug_count}_info.txt")
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(f"URL: {image_url}\n")
            await f.write(f"Expected hash: {expected_hash}\n")
            await f.write(f"Calculated hash: {calculated_hash}\n")
            await f.write(f"File size: {len(image_data)} bytes\n")
            await f.write(f"First 100 bytes (hex): {image_data[:100].hex()}\n")
    
    async def download_image_async(self, session: aiohttp.ClientSession, item: Tuple[int, Dict],
                                  images_dir: str, pbar: tqdm) -> Optional[str]:
        """
        Async function to download a single image with comprehensive error handling
        
        Args:
            session: aiohttp client session
            item: Tuple of (index, example_data)
            images_dir: Directory to save images
            pbar: Progress bar
            
        Returns:
            Calculated hash if successful, None if failed
        """
        if self.graceful_exit.exit_requested:
            return None
            
        i, example = item
        await self.stats.increment('total')
        
        try:
            image_url = example["image_url"]
            expected_hash = example["image_sha256"]
            
            # Check if file already exists
            expected_file_path = os.path.join(images_dir, f"{expected_hash}.png")
            if os.path.exists(expected_file_path):
                await self.stats.increment('skipped_existing')
                pbar.update(1)
                pbar.set_description(f"Skipped existing | Success rate: {self.stats.get_success_rate():.1f}%")
                async with self.calculated_hashes_lock:
                    self.calculated_hashes[i] = expected_hash
                return expected_hash
            
            # Download with retries and exponential backoff
            retry_delays = [1, 2, 4]
            for retry_count in range(self.config.max_retries):
                if self.graceful_exit.exit_requested:
                    return None
                    
                try:
                    if retry_count > 0:
                        delay = retry_delays[min(retry_count-1, len(retry_delays)-1)]
                        await asyncio.sleep(delay)
                        self.log_progress(f"Retry {retry_count}/{self.config.max_retries} for {image_url}")
                    
                    # Stream download for memory efficiency
                    timeout = aiohttp.ClientTimeout(total=self.config.timeout)
                    async with session.get(image_url, timeout=timeout) as response:
                        response.raise_for_status()
                        
                        # Read all content at once
                        image_bytes = await response.read()
                        
                        # Calculate hash
                        calculated_hash = sha256(image_bytes).hexdigest()
                        
                        # Store calculated hash
                        async with self.calculated_hashes_lock:
                            self.calculated_hashes[i] = calculated_hash
                        
                        # Determine filename
                        final_file_path = os.path.join(images_dir, f"{calculated_hash}.png")
                        
                        # Hash verification and logging
                        if calculated_hash != expected_hash:
                            self.log_progress(f"Hash mismatch for {image_url}")
                            self.log_progress(f"Expected: {expected_hash}, Got: {calculated_hash}")
                            
                            # Save debug info for first few mismatches
                            if self.config.save_failed_hash_samples and self.stats.hash_debug_count < 5:
                                await self.save_debug_info(image_bytes, expected_hash, calculated_hash, image_url)
                                self.stats.hash_debug_count += 1
                            
                            await self.stats.increment('hash_mismatch')
                        else:
                            await self.stats.increment('download_success')
                        
                        # Save image atomically
                        temp_path = f"{final_file_path}.tmp"
                        async with aiofiles.open(temp_path, 'wb') as f:
                            await f.write(image_bytes)
                        
                        # Atomic rename
                        os.rename(temp_path, final_file_path)
                        
                        pbar.update(1)
                        pbar.set_description(f"Downloaded | Success rate: {self.stats.get_success_rate():.1f}% | {self.stats.get_throughput():.1f} img/s")
                        
                        return calculated_hash
                        
                except asyncio.TimeoutError:
                    self.log_progress(f"Timeout downloading {image_url} (attempt {retry_count + 1})")
                except aiohttp.ClientError as e:
                    self.log_progress(f"Client error downloading {image_url} (attempt {retry_count + 1}): {str(e)}")
                except Exception as e:
                    self.log_progress(f"Unexpected error downloading {image_url} (attempt {retry_count + 1}): {str(e)}")
            
            # All retries failed
            await self.stats.increment('download_failed')
            self.log_progress(f"Failed to download {image_url} after {self.config.max_retries} attempts", "error")
            pbar.update(1)
            return None
            
        except Exception as e:
            self.log_progress(f"Unexpected error processing item {i}: {str(e)}", "error")
            await self.stats.increment('download_failed')
            pbar.update(1)
            return None
    
    async def worker(self, session: aiohttp.ClientSession, queue: asyncio.Queue, 
                    images_dir: str, pbar: tqdm, worker_id: int):
        """Worker coroutine that processes items from the queue"""
        self.log_progress(f"Worker {worker_id} started")
        while True:
            try:
                # Get item from queue with timeout
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
                if item is None:  # Sentinel value to stop worker
                    queue.task_done()
                    break
                    
                await self.download_image_async(session, item, images_dir, pbar)
                queue.task_done()
                
            except asyncio.TimeoutError:
                # Check if we should exit
                if self.graceful_exit.exit_requested:
                    break
                continue
            except Exception as e:
                self.log_progress(f"Worker {worker_id} error: {str(e)}", "error")
                queue.task_done()
        
        self.log_progress(f"Worker {worker_id} finished")
    
    async def producer(self, data, queue: asyncio.Queue, pbar: tqdm):
        """Producer coroutine that feeds items to the queue"""
        self.log_progress("Producer started")
        for i, example in enumerate(data):
            if self.graceful_exit.exit_requested:
                break
            await queue.put((i, example))
            
            # Log progress every 10000 items
            if (i + 1) % 10000 == 0:
                self.log_progress(f"Queued {i + 1} items")
        
        # Send sentinel values to stop workers
        for _ in range(self.config.max_concurrent):
            await queue.put(None)
        
        self.log_progress("Producer finished")
    
    async def download_images(self, dataset) -> bool:
        """
        Download all images using async producer-consumer pattern
        
        Args:
            dataset: The pixmo points dataset
            
        Returns:
            True if successful, False otherwise
        """
        # Create images directory
        images_dir = os.path.join(self.config.output_path, "images")
        safe_makedirs(images_dir)
        
        # Log configuration
        self.log_progress(f"Hash verification: {'ENABLED' if self.config.verify_hash else 'DISABLED'}")
        self.log_progress(f"Debug mode: {'ENABLED' if self.config.debug_mode else 'DISABLED'}")
        self.log_progress(f"Starting download of {len(dataset)} images with {self.config.max_concurrent} concurrent workers")
        
        # Configure connection limits and timeouts
        connector = aiohttp.TCPConnector(
            limit=self.config.connection_limit,
            limit_per_host=self.config.connection_limit_per_host,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # Configure session timeout
        timeout = aiohttp.ClientTimeout(total=self.config.timeout, connect=10)
        
        # Create queue for producer-consumer pattern
        queue = asyncio.Queue(maxsize=self.config.queue_size)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={},  # Empty headers to avoid content negotiation
            trust_env=True
        ) as session:
            
            # Create progress bar
            with tqdm(total=len(dataset), desc="Downloading images", unit="img") as pbar:
                # Start workers
                workers = [
                    asyncio.create_task(self.worker(session, queue, images_dir, pbar, i))
                    for i in range(self.config.max_concurrent)
                ]
                
                # Start producer
                producer_task = asyncio.create_task(self.producer(dataset, queue, pbar))
                
                try:
                    # Wait for producer to finish
                    await producer_task
                    
                    # Wait for all items to be processed
                    await queue.join()
                    
                    # Wait for all workers to finish
                    await asyncio.gather(*workers, return_exceptions=True)
                    
                except KeyboardInterrupt:
                    self.log_progress("Download interrupted by user")
                    self.graceful_exit.request_exit(None, None)
                    
                    # Cancel all tasks
                    producer_task.cancel()
                    for worker_task in workers:
                        worker_task.cancel()
                    
                    # Wait for cancellation
                    await asyncio.gather(producer_task, *workers, return_exceptions=True)
        
        # Log final statistics
        self.log_progress("Download completed!")
        self.log_progress(f"Final statistics: {self.stats}")
        
        return True
    
    def points_to_text_xml(self, points: np.ndarray, scale: Tuple[float, float], 
                          label_text: str, alt_text: str) -> str:
        """Convert points to XML format"""
        if isinstance(scale, (tuple, list)):
            points = points / np.array(scale)[None, :]
        else:
            points = points * (100/scale)
        
        points = [[int(round(x)), int(round(y))] for x, y in points]
        points.sort(key=lambda x: x[0]*10000 + x[1])
        
        if len(points) == 1:
            x_str, y_str = points[0]
            return f"<point x=\"{x_str}\" y=\"{y_str}\" alt=\"{alt_text}\">{label_text}</point>"
        
        point_text = []
        for ix, (x, y) in enumerate(points, start=1):
            point_text.append(f"x{ix}=\"{x}\"")
            point_text.append(f"y{ix}=\"{y}\"")
        point_text = " ".join(point_text)
        return f"<points {point_text} alt=\"{alt_text}\">{label_text}</points>"
    
    def points_to_text_json(self, points: np.ndarray, scale: Tuple[float, float], 
                           label_text: str, alt_text: str) -> str:
        """Convert points to JSON format"""
        if isinstance(scale, (tuple, list)):
            points = points / np.array(scale)[None, :]
        else:
            points = points * (100/scale)
        
        points = [[int(round(x)), int(round(y))] for x, y in points]
        points.sort(key=lambda x: x[0]*10000 + x[1])
        
        if len(points) == 1:
            x_str, y_str = points[0]
            return f"{{\"x\": {x_str}, \"y\": {y_str}}}"
        
        point_text = []
        for ix, (x, y) in enumerate(points, start=1):
            point_text.append(f"{{\"x\": {x}, \"y\": {y}}}")
        point_text = ", ".join(point_text)
        return f"[{point_text}]"
    
    def points_to_text_list(self, points: np.ndarray, scale: Tuple[float, float], 
                           label_text: str, alt_text: str) -> str:
        """Convert points to coordinate list format"""
        if isinstance(scale, (tuple, list)):
            points = points / np.array(scale)[None, :]
        else:
            points = points * (100/scale)
        
        points = [[int(round(x)), int(round(y))] for x, y in points]
        points.sort(key=lambda x: x[0]*10000 + x[1])
        return f"[{', '.join([f'[{x}, {y}]' for x, y in points])}]"
    
    def convert_points_to_llamafactory_format(self, dataset_with_hash) -> List[Dict[str, Any]]:
        """
        Convert dataset to LLaMA-Factory format
        
        Args:
            dataset_with_hash: Dataset with calculated_hash field
            
        Returns:
            List of converted data items
        """
        self.log_progress("Converting points to LLaMA-Factory format...")
        data_items = []
        images_dir = os.path.join(self.config.output_path, "images")
        
        for data in tqdm(dataset_with_hash, desc="Converting to LLaMA-Factory format"):
            calculated_hash = data["calculated_hash"]
            expected_hash = data["image_sha256"]
            
            if not calculated_hash:
                continue
            if expected_hash != calculated_hash:
                continue
            
            # Verify image exists and is valid
            image_path = os.path.join(images_dir, f"{calculated_hash}.png")
            if not os.path.exists(image_path):
                continue
                
            try:
                image = Image.open(image_path)
                image.verify()
            except Exception as e:
                self.log_progress(f"Image is corrupted: {image_path}", "warning")
                continue
            
            # Get image dimensions for scaling
            w, h = image.size
            h_resized, w_resized = smart_resize(h, w)
            
            relative_image_path = os.path.join(os.path.basename(self.config.output_path), "images", f"{calculated_hash}.png")
            
            # Generate messages for different task types
            task_messages = self.generate_messages(data, (w, h, w_resized, h_resized))
            
            for task_data in task_messages:
                data_item = {
                    "messages": task_data["messages"],
                    "images": [relative_image_path],
                    "id": task_data["id"],
                }
                data_items.append(data_item)
        
        return data_items
    
    def generate_messages(self, data: Dict[str, Any], image_info: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Generate messages for different types of tasks (pointing, counting, QA)
        
        Args:
            data: Dictionary containing the data (points, label, question, answer, etc.)
            image_info: Tuple of (original_width, original_height, resized_width, resized_height)
            
        Returns:
            List of task data dictionaries with messages and task IDs
        """
        w, h, w_resized, h_resized = image_info
        task_messages = []
        
        # Check data type and generate appropriate messages
        if "points" in data and "label" in data:
            # Pointing task data
            task_messages.extend(self._generate_pointing_messages(data, w_resized, h_resized))
        elif "question" in data and "answer" in data:
            # QA task data
            task_messages.extend(self._generate_qa_messages(data))
        elif "conversations" in data:
            # Conversation format data
            task_messages.extend(self._generate_conversation_messages(data))
        else:
            # Try to infer from available fields
            self.log_progress(f"Unknown data format: {list(data.keys())}", "warning")
        
        return task_messages
    
    def _generate_pointing_messages(self, data: Dict[str, Any], w_resized: int, h_resized: int) -> List[Dict[str, Any]]:
        """Generate messages for pointing tasks"""
        messages = []
        points = data["points"]
        label = data["label"]
        
        if len(points) < 1:
            # No points case
            prompt_type = "pointing_json" if np.random.rand() < 0.5 else "pointing_xml"
            prompt = GENERAL_PROMPTS[prompt_type][0]
            prompt = prompt.format(label=label)
            points_text = "This isn't in the image."
        else:
            # Convert points to numpy array
            points_array = np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1)
            
            # Random choice from conversion functions
            convert_func = np.random.choice([self.points_to_text_xml, self.points_to_text_json])
            points_text = convert_func(points_array.copy(), (100 / w_resized, 100 / h_resized), label, label)
            
            # Choose appropriate prompt type
            prompt_type = "pointing_xml" if convert_func == self.points_to_text_xml else "pointing_json"
            prompt = GENERAL_PROMPTS[prompt_type][np.random.randint(0, len(GENERAL_PROMPTS[prompt_type]))]
            prompt = prompt.format(label=label)
        
        # Create main pointing task
        pointing_messages = [
            {"content": "<image>" + prompt, "role": "user"},
            {"content": points_text, "role": "assistant"}
        ]
        
        messages.append({
            "messages": pointing_messages,
            "id": "pointing"
        })
        
        # Add counting task occasionally
        if np.random.rand() < 0.2 and len(points) > 0:
            points_array = np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1)
            prompt_type = "point_count" if np.random.rand() < 0.5 else "count_then_point"
            prompt = GENERAL_PROMPTS[prompt_type][np.random.randint(0, len(GENERAL_PROMPTS[prompt_type]))]
            prompt = prompt.format(label=label)
            
            if prompt_type == "point_count":
                points_text = f"Counting the {label} shows a total of {len(points)}."
            else:
                points_text = f"Counting the {label} shows a total of {len(points)}. " + \
                             self.points_to_text_xml(points_array.copy(), (100 / w_resized, 100 / h_resized), label, label)
            
            counting_messages = [
                {"content": "<image>" + prompt, "role": "user"},
                {"content": points_text, "role": "assistant"}
            ]
            
            messages.append({
                "messages": counting_messages,
                "id": prompt_type
            })
        
        return messages
    
    def _generate_qa_messages(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate messages for QA tasks"""
        question = data["question"]
        answer = data["answer"]
        
        qa_messages = [
            {"content": "<image>" + question, "role": "user"},
            {"content": answer, "role": "assistant"}
        ]
        
        return [{
            "messages": qa_messages,
            "id": "pixmo-ask-anything"
        }]
    
    def _generate_conversation_messages(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate messages for conversation format data"""
        conversations = data["conversations"]
        
        # Convert conversations to messages format
        messages = []
        for conv in conversations:
            if "from" in conv and "value" in conv:
                role = "user" if conv["from"] in ["human", "user"] else "assistant"
                content = conv["value"]
                
                # Add <image> token to first user message if not present
                if role == "user" and len(messages) == 0 and "<image>" not in content:
                    content = "<image>" + content
                
                messages.append({
                    "content": content,
                    "role": role
                })
        
        if messages:
            return [{
                "messages": messages,
                "id": "conversation"
            }]
        else:
            return []
    
    async def process_dataset_async(self) -> bool:
        """
        Main async processing function
        
        Returns:
            True if successful, False otherwise
        """
        if not self.validate_config():
            return False
        
        try:
            # Load dataset
            self.log_progress(f"Loading dataset from {self.config.input_path}")
            dataset = load_dataset(self.config.input_path, split="train")
            
            if dataset is None:
                self.log_progress("Failed to load dataset", "error")
                return False
            
            # if exist calculated_hash, skip download images
            if not self.config.skip_existing:
                # Download images
                if not await self.download_images(dataset):
                    return False
                
                # Create updated dataset with calculated_hash field
                self.log_progress("Creating updated dataset with calculated_hash field...")
                updated_data = []
                for i, example in enumerate(dataset):
                    updated_example = dict(example)
                    updated_example['calculated_hash'] = self.calculated_hashes.get(i, "")
                    updated_data.append(updated_example)
                
                # Save updated dataset
                self.log_progress("Saving updated dataset...")
                updated_dataset = Dataset.from_list(updated_data)
                dataset_output_path = os.path.join(self.config.output_path, f"{self.config.name}-with-calculated-hash")
                updated_dataset.save_to_disk(dataset_output_path)
                self.log_progress(f"Updated dataset saved to: {dataset_output_path}")
            else:
                # Load updated dataset from disk
                dataset_output_path = os.path.join(self.config.output_path, f"{self.config.name}-with-calculated-hash")
                updated_data = Dataset.load_from_disk(dataset_output_path)
            
            # Convert to LLaMA-Factory format
            converted_data = self.convert_points_to_llamafactory_format(updated_data)
            
            # Save final output
            if self.config.output_jsonl:
                output_file = os.path.join(self.config.output_path, f"{self.config.name}-llama-factory.jsonl")
                if save_jsonl_data(converted_data, output_file):
                    self.log_progress(f"LLaMA-Factory format data saved to {output_file}")
                else:
                    return False
            else:
                output_file = os.path.join(self.config.output_path, f"{self.config.name}-llama-factory.json") 
                if save_json_data(converted_data, output_file):
                    self.log_progress(f"LLaMA-Factory format data saved to {output_file}")
                else:
                    return False
            
            # Save final progress
            self._save_progress()
            
            # Log statistics
            total_processed = len([h for h in self.calculated_hashes.values() if h])
            hash_matches = len([i for i, h in self.calculated_hashes.items() 
                              if h == dataset[i]['image_sha256']])
            hash_mismatches = total_processed - hash_matches
            
            self.log_progress(f"Hash verification summary:")
            self.log_progress(f"  Total processed: {total_processed}")
            self.log_progress(f"  Hash matches: {hash_matches}")
            self.log_progress(f"  Hash mismatches: {hash_mismatches}")
            self.log_progress(f"  Match rate: {(hash_matches/max(1, total_processed)*100):.2f}%")
            
            self.log_progress(f"Generated {len(converted_data)} training samples")
            
            return True
            
        except Exception as e:
            self.log_progress(f"Error during dataset processing: {str(e)}", "error")
            return False
    
    def process_dataset(self) -> bool:
        """
        Synchronous wrapper for async processing
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return asyncio.run(self.process_dataset_async())
        except KeyboardInterrupt:
            self.log_progress("Program interrupted")
            return False
        except Exception as e:
            self.log_progress(f"Unexpected error: {str(e)}", "error")
            return False


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = setup_argparse("Pixmo-Points")
    parser.add_argument('--name', type=str, default="pixmo-points",
                       help='Name of the dataset')
    parser.add_argument('--max-concurrent', type=int, default=100,
                       help='Maximum concurrent downloads')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Download timeout in seconds')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts for downloads')
    parser.add_argument('--verify-hash', action='store_true',
                       help='Enable hash verification for downloads')
    parser.add_argument('--debug-mode', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--save-failed-hash-samples', action='store_true',
                       help='Save samples of failed hash verifications')
    parser.add_argument('--output-jsonl', action='store_true',
                       help='Output in JSONL format instead of JSON')
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
    if hasattr(args, 'max_concurrent'):
        config_overrides['max_concurrent'] = args.max_concurrent
    if hasattr(args, 'timeout'):
        config_overrides['timeout'] = args.timeout
    if hasattr(args, 'max_retries'):
        config_overrides['max_retries'] = args.max_retries
    if hasattr(args, 'verify_hash'):
        config_overrides['verify_hash'] = args.verify_hash
    if hasattr(args, 'debug_mode'):
        config_overrides['debug_mode'] = args.debug_mode
    if hasattr(args, 'save_failed_hash_samples'):
        config_overrides['save_failed_hash_samples'] = args.save_failed_hash_samples
    if hasattr(args, 'output_jsonl'):
        config_overrides['output_jsonl'] = args.output_jsonl
    if args.skip_existing:
        config_overrides['skip_existing'] = args.skip_existing
    
    config = config_manager.get_dataset_config("pixmo", **config_overrides)
    
    # Create and run parser
    parser = PixmoParser(config)
    success = parser.process_dataset()
    
    if success:
        print("Pixmo Points dataset processing completed successfully!")
        exit(0)
    else:
        print("Pixmo Points dataset processing failed!")
        exit(1)


if __name__ == "__main__":
    main() 