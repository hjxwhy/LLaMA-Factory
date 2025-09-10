from datasets import load_dataset
import webdataset as wds
import os
import glob
import pickle
import json
from typing import Iterator, Dict, Any, Union, List
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
import random

def decode_rle_mask(rle_data: Dict[str, Any]) -> np.ndarray:
    """
    Decode RLE (Run Length Encoding) mask data to a binary mask image using pycocotools.
    
    Args:
        rle_data: Dictionary containing 'counts' and 'size' keys
        
    Returns:
        Binary mask as numpy array
    """
    try:
        # Use pycocotools to decode the RLE mask
        mask = maskUtils.decode(rle_data)
        return mask.astype(np.uint8)
    except Exception as e:
        # Fallback: create empty mask on decode error
        size = rle_data.get('size', [640, 480])
        return np.zeros((size[1], size[0]), dtype=np.uint8)


def short_side_resize(img: Image.Image, image_size: int, mode: Image.Resampling = Image.Resampling.NEAREST):
        w, h = img.size
        if w < h:
            new_w = image_size
            new_h = int(h * image_size / w)
        else:
            new_h = image_size
            new_w = int(w * image_size / h)
        img = img.resize((new_w, new_h), resample=mode)
        return img

def resize_to_32_multiple(img: Image.Image, mode: Image.Resampling = Image.Resampling.BICUBIC):
    """
    Resize image along the short side to make it a multiple of 32.
    
    Args:
        img: PIL Image object
        mode: Resampling mode
        
    Returns:
        Resized PIL Image
    """
    w, h = img.size
    short_side = min(w, h)
    
    # Find the largest multiple of 32 that's <= short_side
    target_short_side = (short_side // 32) * 32
    if target_short_side == 0:
        target_short_side = 32
    
    # Calculate new dimensions
    if w < h:
        new_w = target_short_side
        new_h = int(h * target_short_side / w)
    else:
        new_h = target_short_side
        new_w = int(w * target_short_side / h)
    
    return img.resize((new_w, new_h), resample=mode)

def extract_directory_structure(local_path: str) -> Dict[str, str]:
    """
    Extract directory structure from __local_path__ for organized saving.
    
    Args:
        local_path: Path like "/jfs/qwen_models/describe-anything-dataset/SAV/images/00000113.tar"
        
    Returns:
        Dict with directory information
    """
    if not local_path:
        return {'parent_dir': '', 'dataset_subdir': '', 'full_subdir': ''}
    
    # Remove file extension and get directory path
    dir_path = os.path.dirname(local_path)
    
    # Split path and find the dataset-specific part
    path_parts = dir_path.split(os.sep)
    
    # Find the base dataset directory index
    dataset_base_idx = -1
    for i, part in enumerate(path_parts):
        if 'describe-anything-dataset' in part:
            dataset_base_idx = i
            break
    
    if dataset_base_idx == -1:
        # Fallback: use last two directories
        if len(path_parts) >= 2:
            parent_dir = path_parts[-2]
            dataset_subdir = path_parts[-1]
            full_subdir = os.path.join(parent_dir, dataset_subdir)
        else:
            parent_dir = path_parts[-1] if path_parts else ''
            dataset_subdir = ''
            full_subdir = parent_dir
    else:
        # Extract directories after the dataset base
        remaining_parts = path_parts[dataset_base_idx + 1:]
        if len(remaining_parts) >= 2:
            parent_dir = remaining_parts[0]  # e.g., "SAV"
            dataset_subdir = remaining_parts[1]  # e.g., "images"
            full_subdir = os.path.join(*remaining_parts)  # e.g., "SAV/images"
        elif len(remaining_parts) == 1:
            parent_dir = remaining_parts[0]
            dataset_subdir = ''
            full_subdir = parent_dir
        else:
            parent_dir = ''
            dataset_subdir = ''
            full_subdir = ''
    
    return {
        'parent_dir': parent_dir,
        'dataset_subdir': dataset_subdir,
        'full_subdir': full_subdir,
        'original_path': local_path
    }

def process_single_image_mask_pair(image: Image.Image, mask: Image.Image):
    """
    Process a single image-mask pair according to the new requirements:
    1. Resize along short side to 32-pixel multiple
    2. Smart square crop to maximize mask inclusion (if mask exists)
    3. Scale down to 512 if larger
    4. Prepare mask for VAE (128x128) if mask exists
    
    Args:
        image: PIL Image object
        mask: PIL Image object (can be None)
        
    Returns:
        Dict containing processed image, mask, and VAE mask
    """
    if image is None:
        return None
    
    # Step 1: Resize to 32-pixel multiple along short side
    image = resize_to_32_multiple(image, Image.Resampling.BICUBIC)
    if mask is not None:
        mask = resize_to_32_multiple(mask, Image.Resampling.NEAREST)
    
    # Step 2: Smart square crop
    # Smart crop with mask to maximize mask inclusion
    image, mask = smart_square_crop_with_mask(image, mask)

    # Step 3: Scale down to 512 if larger than 512
    if max(image.size) > 512:
        scale_factor = 512 / max(image.size)
        new_w = int(image.size[0] * scale_factor)
        new_h = int(image.size[1] * scale_factor)
        
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        if mask is not None:
            mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
    
    # Step 4: Prepare mask for VAE (128x128) if mask exists
    vae_mask = None
    if mask is not None:
        vae_mask = mask.resize((128, 128), Image.Resampling.NEAREST)
    
    return {
        'image': image,
        'mask': mask,
        'vae_mask': vae_mask,
        'final_size': image.size
    }

def save_processed_image_and_mask(sample_data: Dict[str, Any], save_base_dir: str = "/data0/data", save_mask=False):
    """
    Save processed image and mask to organized directory structure.
    
    Args:
        sample_data: Processed sample data
        save_base_dir: Base directory for saving
        
    Returns:
        Dict with saved file paths
    """
    if not sample_data.get('image'):
        return None
    
    # Create directory structure
    full_save_dir = os.path.join(save_base_dir, sample_data.get('full_subdir', ''))
    os.makedirs(full_save_dir, exist_ok=True)
    
    # Prepare file paths
    image_filename = sample_data.get('save_name', f"{sample_data.get('key', 'unknown')}.jpg")
    image_save_path = os.path.join(full_save_dir, image_filename)
    
    # Save image
    # try:
    sample_data['image'].save(image_save_path, 'JPEG', quality=95)
    saved_paths = {'image_path': image_save_path}
    
    if save_mask:
        # Save mask if available
        if sample_data.get('mask'):
            mask_filename = image_filename.replace('.jpg', '_mask.png')
            mask_save_path = os.path.join(full_save_dir, mask_filename)
            sample_data['mask'].save(mask_save_path, 'PNG')
            saved_paths['mask_path'] = mask_save_path
        
        # Save VAE mask if available
        if sample_data.get('vae_mask'):
            vae_mask_filename = image_filename.replace('.jpg', '_vae_mask.png')
            vae_mask_save_path = os.path.join(full_save_dir, vae_mask_filename)
            sample_data['vae_mask'].save(vae_mask_save_path, 'PNG')
            saved_paths['vae_mask_path'] = vae_mask_save_path
    
    return saved_paths
        
    # except Exception as e:
    #     # Silent error handling for file saving
    #     return None

def create_conversation_data(sample_data_list: List[Dict[str, Any]], relative_image_paths: List[str] = None):
    """
    Create conversation format data for training.
    
    Args:
        sample_data_list: List of processed sample data (single item for single image, multiple for multi-image)
        relative_image_paths: List of relative paths to the images for the conversation
        
    Returns:
        Dict containing conversation format
    """
    if not sample_data_list:
        return None
    
    # For single image case, just use the first item
    if len(sample_data_list) == 1:
        sample_data = sample_data_list[0]
        image_path = relative_image_paths[0] if relative_image_paths else sample_data.get('save_path', sample_data.get('save_name', ''))
        
        # Get caption, fallback to default if not available
        caption = sample_data.get('caption', 'Describe this image and provide segmentation masks.')
        
        # Clean caption by removing video/image tags
        caption = caption.replace("<video>", "").replace("<image>", "").strip()
        
        # Create the conversation
        default_prompt = "<image>" + "According the descritions, give the segmentation masks. Descritions:"
        user_content = default_prompt + "\n" + caption
        
        # Get segmentation tokens
        seg_tokens = sample_data.get('seg_token_string', '')
        
        # If no segmentation tokens, use a placeholder message
        assistant_content = seg_tokens if seg_tokens else "I can see the image but cannot provide segmentation masks."
        
        conversation = {
            "messages": [
                {
                    "role": "user", 
                    "content": user_content
                },
                {
                    "role": "assistant", 
                    "content": assistant_content
                }
            ],
            "images": [image_path]
        }
        
        # Add metadata
        conversation["metadata"] = {
            "key": sample_data.get('key'),
            "category": sample_data.get('category'),
            "image_size": sample_data.get('final_size'),
            "is_single_image": True,
            "parent_dir": sample_data.get('parent_dir'),
            "full_subdir": sample_data.get('full_subdir')
        }
        
    else:
        # Multi-image case: combine all images in one conversation
        first_sample = sample_data_list[0]
        
        # Use images paths
        image_paths = relative_image_paths if relative_image_paths else [s.get('save_path', s.get('save_name', '')) for s in sample_data_list]
        
        # Get caption from first sample (they should all have the same caption for multi-image)
        caption = first_sample.get('caption', 'Describe these images and provide segmentation masks.')
        caption = caption.replace("<video>", "").replace("<image>", "").strip()
        
        # Create multi-image prompt with multiple <image> tags
        image_tags = "".join(["<image>" for _ in range(len(sample_data_list))])
        default_prompt = image_tags + "According the descritions, give the segmentation masks. Descritions:"
        user_content = default_prompt + "\n" + caption
        
        # Combine all segmentation tokens
        all_seg_tokens = []
        for sample in sample_data_list:
            seg_token = sample.get('seg_token_string', '')
            if seg_token:
                all_seg_tokens.append(seg_token)
        
        combined_seg_tokens = "".join(all_seg_tokens)
        
        # If no segmentation tokens, use a placeholder message
        assistant_content = combined_seg_tokens if combined_seg_tokens else "I can see the images but cannot provide segmentation masks."
        
        conversation = {
            "messages": [
                {
                    "role": "user", 
                    "content": user_content
                },
                {
                    "role": "assistant", 
                    "content": assistant_content
                }
            ],
            "images": image_paths
        }
        
        # Add metadata
        conversation["metadata"] = {
            "key": first_sample.get('key'),
            "category": first_sample.get('category'),
            "image_count": len(sample_data_list),
            "image_sizes": [s.get('final_size') for s in sample_data_list],
            "is_single_image": False,
            "parent_dir": first_sample.get('parent_dir'),
            "full_subdir": first_sample.get('full_subdir'),
            "image_indices": [s.get('image_index') for s in sample_data_list]
        }
    
    return conversation

def save_conversations_to_json(conversations: List[Dict[str, Any]], save_path: str):
    """
    Save conversations to JSON file.
    
    Args:
        conversations: List of conversation dictionaries
        save_path: Path to save the JSON file
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        return False

def smart_square_crop_with_mask(img: Image.Image, mask: Image.Image):
    """
    Smart cropping to create a square image that maximizes mask inclusion.
    The square size is based on the shorter side of the resized image.
    
    Args:
        img: PIL Image object  
        mask: PIL Image object (mask)
        
    Returns:
        Tuple of (cropped_img, cropped_mask)
    """
    img_w, img_h = img.size
    
    # Calculate square size based on shorter side
    square_size = min(img_w, img_h)
    
    # Convert mask to numpy for processing
    mask_array = np.array(mask)
    
    # Find mask bounding box
    mask_indices = np.where(mask_array > 0)
    if len(mask_indices[0]) == 0:
        # No valid mask region, return center crop
        left = (img_w - square_size) // 2
        top = (img_h - square_size) // 2
        right = left + square_size
        bottom = top + square_size
        
        cropped_img = img.crop((left, top, right, bottom))
        cropped_mask = mask.crop((left, top, right, bottom))
        return cropped_img, cropped_mask
    
    min_y, max_y = mask_indices[0].min(), mask_indices[0].max()
    min_x, max_x = mask_indices[1].min(), mask_indices[1].max()
    
    # Calculate mask center
    mask_center_y = (min_y + max_y) // 2
    mask_center_x = (min_x + max_x) // 2
    
    # Calculate mask dimensions
    mask_width = max_x - min_x + 1
    mask_height = max_y - min_y + 1
    
    # Try different cropping strategies to maximize mask inclusion
    best_crop = None
    best_mask_ratio = 0
    
    # Strategy 1: Center crop around mask center
    crop_left = max(0, min(img_w - square_size, mask_center_x - square_size // 2))
    crop_top = max(0, min(img_h - square_size, mask_center_y - square_size // 2))
    crop_right = crop_left + square_size
    crop_bottom = crop_top + square_size
    
    # Calculate mask ratio for this crop
    cropped_mask_array = mask_array[crop_top:crop_bottom, crop_left:crop_right]
    total_pixels = cropped_mask_array.size
    valid_pixels = np.sum(cropped_mask_array > 0)
    mask_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0
    
    if mask_ratio > best_mask_ratio:
        best_mask_ratio = mask_ratio
        best_crop = (crop_left, crop_top, crop_right, crop_bottom)
    
    # Strategy 2: Try a few random offsets if mask ratio is low
    if best_mask_ratio < 0.3:
        for _ in range(5):
            offset_x = random.randint(-square_size//4, square_size//4)
            offset_y = random.randint(-square_size//4, square_size//4)
            
            crop_left = max(0, min(img_w - square_size, mask_center_x - square_size // 2 + offset_x))
            crop_top = max(0, min(img_h - square_size, mask_center_y - square_size // 2 + offset_y))
            crop_right = crop_left + square_size
            crop_bottom = crop_top + square_size
            
            cropped_mask_array = mask_array[crop_top:crop_bottom, crop_left:crop_right]
            total_pixels = cropped_mask_array.size
            valid_pixels = np.sum(cropped_mask_array > 0)
            mask_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0
            
            if mask_ratio > best_mask_ratio:
                best_mask_ratio = mask_ratio
                best_crop = (crop_left, crop_top, crop_right, crop_bottom)
    
    # Use the best crop found
    if best_crop is not None:
        crop_left, crop_top, crop_right, crop_bottom = best_crop
    else:
        # Fallback to center crop
        crop_left = (img_w - square_size) // 2
        crop_top = (img_h - square_size) // 2
        crop_right = crop_left + square_size
        crop_bottom = crop_top + square_size
    
    cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    cropped_mask = mask.crop((crop_left, crop_top, crop_right, crop_bottom))
    
    return cropped_img, cropped_mask


def enhanced_load_sample_to_dict(sample: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Enhanced version that handles both single and multi-image cases.
    
    Args:
        sample: Raw sample from webdataset
        
    Returns:
        Single dict for single image or list of dicts for multi-image
    """
    base_key = sample.get('__key__')
    url = sample.get('__url__')
    local_path = sample.get('__local_path__')
    
    # Extract directory structure for organized saving
    dir_info = extract_directory_structure(local_path)
    
    # Detect if single or multi-image
    single_image = "jpg" in sample.keys()
    
    if single_image:
        # Single image case
        result = {
            'key': base_key,
            'url': url,
            'local_path': local_path,
            'image': sample['jpg'],
            'image_size': sample['jpg'].size if sample['jpg'] else None,
            'mask': None,
            'is_single_image': True,
            'save_name': f"{base_key}.jpg",
            'parent_dir': dir_info['parent_dir'],
            'dataset_subdir': dir_info['dataset_subdir'],
            'full_subdir': dir_info['full_subdir'],
            'save_path': os.path.join(dir_info['full_subdir'], f"{base_key}.jpg") if dir_info['full_subdir'] else f"{base_key}.jpg"
        }
        
        # Process pickle data for single image
        if 'pickle' in sample:
            pickle_data = sample['pickle']
            if isinstance(pickle_data, bytes):
                metadata = pickle.loads(pickle_data)
            else:
                metadata = pickle_data

            # breakpoint()
                
            # For single image, metadata should be a dict
            if isinstance(metadata, list) and len(metadata) > 0:
                metadata = metadata[0]
            
            if isinstance(metadata, dict):
                result.update({
                    'img_id': metadata.get('img_id'),
                    'ann_id': metadata.get('ann_id'),
                    'image_path': metadata.get('image'),
                    'category': metadata.get('category'),
                    'caption': metadata.get('caption')
                })
                
                # For single image, mask_rle should be a dict
                mask_rle = metadata.get('mask_rle')
                if mask_rle:
                    if isinstance(mask_rle, list) and len(mask_rle) > 0:
                        mask_rle = mask_rle[0]
                    
                    mask = decode_rle_mask(mask_rle)
                    if isinstance(mask, np.ndarray):
                        mask = mask * 255
                        mask = Image.fromarray(mask, mode='L')
                    result['mask'] = mask
                    result['mask_size'] = mask.size if mask else None
        
        return result
    
    else:
        # Multi-image case
        image_keys = [key for key in sample.keys() if key.endswith('.jpg') and key[:-4].isdigit()]
        if not image_keys:
            return []
        
        image_keys.sort()
        results = []
        
        # Process pickle data first to get metadata list
        metadata_list = []
        if 'pickle' in sample:
            pickle_data = sample['pickle']
            if isinstance(pickle_data, bytes):
                metadata_list = pickle.loads(pickle_data)
            else:
                metadata_list = pickle_data
                
            # breakpoint()
            assert isinstance(metadata_list, dict), "metadata_list is not a dict"
                
            # if isinstance(metadata_list, dict):
            #     metadata_list = [metadata_list]
        
        # Create result for each image
        for i, image_key in enumerate(image_keys):
            image_idx = int(image_key[:-4])
            
            result = {
                'key': base_key,
                'url': url,
                'local_path': local_path,
                'image': sample.get(image_key),
                'image_size': sample[image_key].size if sample.get(image_key) else None,
                'mask': None,
                'is_single_image': False,
                'image_index': image_idx,
                'save_name': f"{base_key}_{image_idx}.jpg",
                'parent_dir': dir_info['parent_dir'],
                'dataset_subdir': dir_info['dataset_subdir'],
                'full_subdir': dir_info['full_subdir'],
                'save_path': os.path.join(dir_info['full_subdir'], f"{base_key}_{image_idx}.jpg") if dir_info['full_subdir'] else f"{base_key}_{image_idx}.jpg"
            }
            
            # Get metadata (usually only one metadata object for all images)
            if metadata_list:
                # Use the first (and usually only) metadata object
                metadata = metadata_list[0] if isinstance(metadata_list, list) else metadata_list
                
                if isinstance(metadata, dict):
                    result.update({
                        'img_id': metadata.get('video_id'),
                        'ann_id': metadata.get('ann_id'),
                        'image_path': metadata.get('image'),
                        'category': metadata.get('category'),
                        'caption': metadata.get('caption')
                    })
                    
                    # For multi-image, mask_rle should be a list with one mask per image
                    mask_rle_list = metadata.get('mask_rle')
                    # if mask_rle_list and isinstance(mask_rle_list, list) and i < len(mask_rle_list):
                    mask_rle = mask_rle_list[i]
                    mask = decode_rle_mask(mask_rle)
                    if isinstance(mask, np.ndarray):
                        mask = mask * 255
                        mask = Image.fromarray(mask, mode='L')
                    result['mask'] = mask
                    result['mask_size'] = mask.size if mask else None
                    # elif mask_rle_list and not isinstance(mask_rle_list, list):
                    #     # Fallback: if mask_rle is not a list, use it directly (only for first image)
                    #     if i == 0:
                    #         mask = decode_rle_mask(mask_rle_list)
                    #         if isinstance(mask, np.ndarray):
                    #             mask = mask * 255
                    #             mask = Image.fromarray(mask, mode='L')
                    #         result['mask'] = mask
                    #         result['mask_size'] = mask.size if mask else None
            
            results.append(result)
        
        return results


def load_webdataset_datasets(data_path: str) -> Iterator[Dict[str, Any]]:
    """
    Load webdataset from tar files.
    
    Args:
        data_path: Path to the directory containing .tar files or pattern for tar files
        
    Returns:
        Iterator over dataset samples
    """

    # Handle different path formats
    if os.path.isdir(data_path):
        # If it's a directory, create pattern for all tar files
        pattern = os.path.join(data_path, "**", "*.tar")
        # Expand the glob pattern to get actual file paths
        tar_files = glob.glob(pattern, recursive=True)
        if not tar_files:
            raise FileNotFoundError(f"No tar files found in {data_path}")
        tar_files.sort()  # Sort for consistent ordering
        urls = tar_files
    else:
        # Assume it's already a pattern or list of files
        if isinstance(data_path, str):
            urls = glob.glob(data_path)
        else:
            urls = data_path
    # Optional: use specific tar files for testing
    # urls = ["/jfs/qwen_models/describe-anything-dataset/SAV/images/00000113.tar", "/jfs/qwen_models/describe-anything-dataset/SAM/images/00000264.tar"]
    
    dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
    return dataset


def load_webdataset(data_path: str, shuffle: bool = True, repeat: bool = True) -> Iterator[Dict[str, Any]]:
    """
    Load webdataset from tar files.
    
    Args:
        data_path: Path to the directory containing .tar files or pattern for tar files
        shuffle: Whether to shuffle the dataset
        repeat: Whether to repeat the dataset infinitely
        
    Returns:
        Iterator over dataset samples
    """
    # Handle different path formats
    if os.path.isdir(data_path):
        # If it's a directory, create pattern for all tar files
        pattern = os.path.join(data_path, "**", "*.tar")
        # Expand the glob pattern to get actual file paths
        tar_files = glob.glob(pattern, recursive=True)
        if not tar_files:
            raise FileNotFoundError(f"No tar files found in {data_path}")
        tar_files.sort()  # Sort for consistent ordering
        urls = tar_files
    else:
        # Assume it's already a pattern or list of files
        if isinstance(data_path, str):
            urls = glob.glob(data_path)
        else:
            urls = data_path
    # Optional: use specific tar files for testing
    # urls = ["/jfs/qwen_models/describe-anything-dataset/SAV/images/00000113.tar"]
    
    # Create webdataset with proper shardshuffle parameter
    dataset = wds.WebDataset(urls, shardshuffle=1000 if shuffle else False)
    
    if shuffle:
        dataset = dataset.shuffle(1000)  # Shuffle with buffer size
    
    if repeat:
        dataset = dataset.repeat()
    
    # Decode common formats
    dataset = dataset.decode("pil")  # For images
    
    return dataset


def create_enhanced_training_dataset(data_path: str, max_samples: int = None, vae_processor=None):
    """
    Create an enhanced training dataset that handles both single and multi-image cases.
    Groups multi-image samples together to maintain their relationship.
    
    Args:
        data_path: Path to the webdataset tar files
        max_samples: Maximum number of samples to process (None for all)
        vae_processor: VAE processor for mask tokenization
    
    Returns:
        Iterator yielding training samples (single dict for single-image, list for multi-image)
    """
    dataset = load_webdataset(data_path, shuffle=False, repeat=False)
    
    count = 0
    for sample in dataset:
        if max_samples and count >= max_samples:
            break
            
        # try:
        # Use enhanced loader that handles both single and multi-image
        structured_data = enhanced_load_sample_to_dict(sample)
        
        # Handle both single result (dict) and multi-image results (list)
        if isinstance(structured_data, dict):
            # Single image case
            structured_samples = [structured_data]
        else:
            # Multi-image case
            structured_samples = structured_data
        
        # Process all images in this sample
        processed_samples = []
        for structured_sample in structured_samples:
            if structured_sample.get('image') is None:
                continue
            
            # Apply enhanced processing pipeline
            processed_data = process_single_image_mask_pair(
                structured_sample['image'], 
                structured_sample.get('mask')
            )
            
            if processed_data is None:
                continue
            
            # Create training sample with directory structure info
            training_sample = {
                'image': processed_data['image'],
                'mask': processed_data['mask'],
                'vae_mask': processed_data['vae_mask'],
                'final_size': processed_data['final_size'],
                'save_name': structured_sample.get('save_name'),
                'save_path': structured_sample.get('save_path'),
                'is_single_image': structured_sample.get('is_single_image', True),
                'image_index': structured_sample.get('image_index'),
                'key': structured_sample.get('key'),
                'image_id': structured_sample.get('img_id'),
                'annotation_id': structured_sample.get('ann_id'),
                'category': structured_sample.get('category'),
                'image_path': structured_sample.get('image_path'),
                'caption': structured_sample.get('caption'),
                'url': structured_sample.get('url'),
                'local_path': structured_sample.get('local_path'),
                'parent_dir': structured_sample.get('parent_dir'),
                'dataset_subdir': structured_sample.get('dataset_subdir'),
                'full_subdir': structured_sample.get('full_subdir')
            }
            
            # Add VAE tokens if processor is available
            if vae_processor is not None and processed_data['vae_mask'] is not None:
                # try:
                # Convert PIL mask to numpy array and normalize
                vae_mask_array = np.array(processed_data['vae_mask']).astype(np.float32) / 255.0
                
                # Generate VAE tokens
                indices = vae_processor.encode(vae_mask_array).cpu().numpy().flatten().tolist()
                seg_tokens = ["<seg{:03d}>".format(i) for i in indices]
                seg_token_str = "".join(seg_tokens)
                seg_token_str = "<seg_begin>" + seg_token_str + "<seg_end>"
                
                training_sample['seg_tokens'] = seg_tokens
                training_sample['seg_token_string'] = seg_token_str
                # except Exception as e:
                #     # Silent error handling for VAE token generation
                #     training_sample['seg_tokens'] = None
                #     training_sample['seg_token_string'] = None
            
            processed_samples.append(training_sample)
        
        # Yield the complete sample group
        if processed_samples:
            # if len(processed_samples) == 1:
            #     # Single image: yield the dict directly
            #     yield processed_samples[0]
            # else:
            #     # Multi-image: yield the list to maintain grouping
            yield processed_samples
            
            count += 1
            
            if max_samples and count >= max_samples:
                return
                    
        # except Exception as e:
        #     # Silent error handling for sample processing
        #     continue



def blend_mask_with_image(image, mask, alpha=0.5, color=[255, 0, 0]):
    """
    Blend mask with image for visualization.
    
    Args:
        image: PIL Image object
        mask: numpy array mask (0 and 1 values)
        alpha: transparency of the mask overlay (0.0 to 1.0)
        color: RGB color for the mask overlay [R, G, B]
        
    Returns:
        PIL Image with mask overlay
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Ensure mask is the same size as image
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    if mask.size != image.size:
        # Resize mask to match image size
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    mask = np.array(mask)
    # Create colored mask overlay
    colored_mask = np.zeros_like(img_array)
    colored_mask[mask > 0] = color
    
    # Blend image with mask
    blended = img_array.copy().astype(np.float32)
    blended[mask > 0] = (1 - alpha) * blended[mask > 0] + alpha * colored_mask[mask > 0]
    
    # Convert back to PIL Image
    return Image.fromarray(blended.astype(np.uint8))


def visualize_sample_with_mask(sample_dict, save_path=None, alpha=0.6, color=[255, 0, 0]):
    """
    Visualize a sample with its mask overlaid on the image.
    
    Args:
        sample_dict: Dictionary containing 'image' and 'mask' keys
        save_path: Optional path to save the visualization
        alpha: Transparency of mask overlay
        color: RGB color for mask overlay
        
    Returns:
        PIL Image with mask overlay
    """
    image = sample_dict.get('image')
    mask = sample_dict.get('mask')
    
    if image is None:
        print("No image found in sample")
        return None
    
    if mask is None:
        print("No mask found in sample")
        return image
    
    # Blend mask with image
    blended_image = blend_mask_with_image(image, mask, alpha, color)
    
    # Save if path provided
    if save_path:
        blended_image.save(save_path)
        print(f"Visualization saved to: {save_path}")
    
    return blended_image


def create_visualization_grid(samples, n_cols=3, save_path="visualization_grid.png"):
    """
    Create a grid visualization of multiple samples with masks.
    
    Args:
        samples: List of sample dictionaries
        n_cols: Number of columns in the grid
        save_path: Path to save the grid image
        
    Returns:
        PIL Image of the grid
    """
    if not samples:
        print("No samples provided")
        return None
    
    # Calculate grid dimensions
    n_samples = len(samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Get dimensions from first sample
    first_sample = samples[0]
    if first_sample.get('image') is None:
        print("No image found in first sample")
        return None
    
    img_width, img_height = first_sample['image'].size
    
    # Create grid canvas
    grid_width = n_cols * img_width
    grid_height = n_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    # Place each sample in the grid
    for i, sample in enumerate(samples):
        if sample.get('image') is None:
            continue
            
        # Calculate position in grid
        row = i // n_cols
        col = i % n_cols
        x = col * img_width
        y = row * img_height
        
        # Create visualization for this sample
        vis_image = visualize_sample_with_mask(sample, alpha=0.6, color=[255, 0, 0])
        if vis_image:
            grid_image.paste(vis_image, (x, y))
    
    # Save grid
    grid_image.save(save_path)
    print(f"Grid visualization saved to: {save_path}")
    
    return grid_image



import sys
sys.path.append("/jfs/jensen/code/LLaMA-Factory")
from src.llamafactory.data.data_utils import get_process_mask_func
from datasets import Dataset, IterableDataset

def process_dataset_samples(dataset_path: str, save_dir: str = "/data0/data/describe-anything-dataset", 
                           max_samples: int = None, save_images: bool = False, save_conversations: bool = False,
                           verbose: bool = False):
    """
    Process dataset samples with enhanced functionality.
    
    Args:
        dataset_path: Path to the webdataset tar files
        save_dir: Directory to save processed images
        max_samples: Maximum number of samples to process
        save_images: Whether to save processed images to disk
        save_conversations: Whether to save conversation data to JSON
        verbose: Whether to print detailed processing information
    
    Returns:
        Tuple of (processed samples, conversations)
    """
    # Load VAE processor for tokenization
    try:
        from src.llamafactory.data.VQVAE import get_vqvae_processor
        vae_processor = get_vqvae_processor()
    except Exception as e:
        vae_processor = None

    assert vae_processor is not None, "VAE processor is not loaded"
    
    # Process training dataset
    enhanced_training_data = create_enhanced_training_dataset(
        dataset_path, 
        max_samples=max_samples, 
        vae_processor=vae_processor
    )
    
    samples_processed = []
    conversations = []
    sample_index = 0
    
    for sample_group in enhanced_training_data:
        # Handle both single samples (dict) and multi-image groups (list)
        if isinstance(sample_group, dict):
            # Single image sample
            sample_list = [sample_group]
        else:
            # Multi-image sample group
            sample_list = sample_group
        
        # Process images and save them
        processed_group = []
        relative_paths = []
        
        for sample in sample_list:
            # Save images if requested
            if save_images:
                saved_paths = save_processed_image_and_mask(sample, save_dir)
                if saved_paths:
                    sample['saved_paths'] = saved_paths
                    relative_image_path = os.path.relpath(saved_paths['image_path'], save_dir)
                    sample['relative_image_path'] = relative_image_path
                    relative_paths.append(relative_image_path)
            
            processed_group.append(sample)
            samples_processed.append(sample)
        
        # Create conversations if requested
        if save_conversations and processed_group:
            # Create conversation for this group (even if no segmentation tokens)
            conversation = create_conversation_data(processed_group, relative_paths)
            if conversation:
                conversations.append(conversation)
        
        sample_index += 1
        if verbose and sample_index % 100 == 0:
            print(f"Processed {sample_index} sample groups...")
    
    # Save conversations to JSON if requested
    if save_conversations and conversations:
        conversation_json_path = os.path.join(save_dir, "conversations.json")
        save_conversations_to_json(conversations, conversation_json_path)
    
    return samples_processed, conversations


if __name__ == "__main__":
    # Configuration
    dataset_path = "/jfs/qwen_models/describe-anything-dataset"
    save_dir = "/data0/data/describe-anything-dataset"
    
    # Process dataset samples
    samples, conversations = process_dataset_samples(
        dataset_path=dataset_path,
        save_dir=save_dir,
        max_samples=None,
        save_images=True,
        save_conversations=True,
        verbose=True
    )
    
    print(f"Processed {len(samples)} samples and created {len(conversations)} conversations.")
    
    # Create visualization if samples exist
    if samples:
        try:
            create_visualization_grid(samples[:3], save_path="sample_visualization.png")
            print("Visualization saved as 'sample_visualization.png'")
        except Exception as e:
            print(f"Could not create visualization: {e}")

