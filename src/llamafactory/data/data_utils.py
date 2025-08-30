# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union

import glob
import pickle
from typing import Iterator, Dict, Any
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
import random

import fsspec
from datasets import DatasetDict, concatenate_datasets, interleave_datasets
from webdataset.compat import WebDataset

from ..extras import logging


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


SLOTS = list[Union[str, set[str], dict[str, str]]]


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]


def merge_dataset(
    all_datasets: list[Union["Dataset", "IterableDataset"]], data_args: "DataArguments", seed: int
) -> Union["Dataset", "IterableDataset"]:
    r"""Merge multiple datasets to a unified dataset."""
    if len(all_datasets) == 1:
        return all_datasets[0]

    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning_rank0_once("The samples between different datasets will not be mixed in streaming mode.")

        return concatenate_datasets(all_datasets)

    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning_rank0_once("We recommend using `mix_strategy=concat` in non-streaming mode.")

        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )

    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")


def split_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]],
    data_args: "DataArguments",
    seed: int,
) -> "DatasetDict":
    r"""Split the dataset and returns a dataset dict containing train set and validation set.

    Support both map dataset and iterable dataset.
    """
    if eval_dataset is not None and data_args.val_size > 1e-6:
        raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

    dataset_dict = {}
    if dataset is not None:
        if data_args.streaming and not isinstance(dataset, WebDataset):
            dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

        if data_args.val_size > 1e-6:
            if data_args.streaming:
                dataset_dict["validation"] = dataset.take(int(data_args.val_size))
                dataset_dict["train"] = dataset.skip(int(data_args.val_size))
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                dataset_dict = dataset.train_test_split(test_size=val_size, seed=seed)
                dataset = dataset.train_test_split(test_size=val_size, seed=seed)
                dataset_dict = {"train": dataset["train"], "validation": dataset["test"]}
        else:
            dataset_dict["train"] = dataset

    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            dataset_dict.update({f"validation_{name}": data for name, data in eval_dataset.items()})
        else:
            if data_args.streaming:
                eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

            dataset_dict["validation"] = eval_dataset

    return DatasetDict(dataset_dict)


def get_dataset_module(dataset: Union["Dataset", "DatasetDict"]) -> "DatasetModule":
    r"""Convert dataset or dataset dict to dataset module."""
    dataset_module: DatasetModule = {}
    if isinstance(dataset, DatasetDict):  # dataset dict
        if "train" in dataset:
            dataset_module["train_dataset"] = dataset["train"]

        if "validation" in dataset:
            dataset_module["eval_dataset"] = dataset["validation"]
        else:
            eval_dataset = {}
            for key in dataset.keys():
                if key.startswith("validation_"):
                    eval_dataset[key[len("validation_") :]] = dataset[key]

            if len(eval_dataset):
                dataset_module["eval_dataset"] = eval_dataset

    else:  # single dataset
        dataset_module["train_dataset"] = dataset

    return dataset_module


def setup_fs(path: str, anon: bool = False) -> "fsspec.AbstractFileSystem":
    r"""Set up a filesystem object based on the path protocol."""
    storage_options = {"anon": anon} if anon else {}
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **storage_options)
    elif path.startswith(("gs://", "gcs://")):
        fs = fsspec.filesystem("gcs", **storage_options)
    else:
        raise ValueError(f"Unsupported protocol in path: {path}. Use 's3://' or 'gs://'.")

    if not fs.exists(path):
        raise ValueError(f"Path does not exist: {path}.")

    return fs


def _read_json_with_fs(fs: "fsspec.AbstractFileSystem", path: str) -> list[Any]:
    r"""Helper function to read JSON/JSONL files using fsspec."""
    with fs.open(path, "r") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


def read_cloud_json(cloud_path: str) -> list[Any]:
    r"""Read a JSON/JSONL file from cloud storage (S3 or GCS).

    Args:
        cloud_path: str
            Cloud path in the format:
            - 's3://bucket-name/file.json' for AWS S3
            - 'gs://bucket-name/file.jsonl' or 'gcs://bucket-name/file.jsonl' for Google Cloud Storage
    """
    try:
        fs = setup_fs(cloud_path, anon=True)  # try with anonymous access first
    except Exception:
        fs = setup_fs(cloud_path)  # try again with credentials

    # filter out non-JSON files
    files = [x["Key"] for x in fs.listdir(cloud_path)] if fs.isdir(cloud_path) else [cloud_path]
    files = filter(lambda file: file.endswith(".json") or file.endswith(".jsonl"), files)
    if not files:
        raise ValueError(f"No JSON/JSONL files found in the specified path: {cloud_path}.")

    return sum([_read_json_with_fs(fs, file) for file in files], [])


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
        print(f"Error decoding RLE mask with pycocotools: {e}")
        # Fallback: create empty mask
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

def crop_image_and_mask(img: Image.Image, mask: Image.Image, image_size: int):
    # 需要根据mask来crop图片和mask, crop出来的mask有效区域（mask>0）需要大于50%，如果小于50%需要重新生成crop区域的边界
    
    img_w, img_h = img.size
    mask_w, mask_h = mask.size
    
    # 将mask转换为numpy数组进行处理
    mask_array = np.array(mask)
    
    # 找到mask的边界框
    mask_indices = np.where(mask_array > 0)
    if len(mask_indices[0]) == 0:
        # 如果没有有效的mask区域，返回中心crop
        crop_size = min(img_w, img_h, image_size)
        left = (img_w - crop_size) // 2
        top = (img_h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        cropped_img = img.crop((left, top, right, bottom))
        cropped_mask = mask.crop((left, top, right, bottom))
        return cropped_img, cropped_mask
    
    min_y, max_y = mask_indices[0].min(), mask_indices[0].max()
    min_x, max_x = mask_indices[1].min(), mask_indices[1].max()
    
    # 计算mask的中心点
    center_y = (min_y + max_y) // 2
    center_x = (min_x + max_x) // 2
    
    # 尝试生成合适的crop区域
    max_attempts = 10
    for attempt in range(max_attempts):
        # 计算crop区域的边界
        half_size = image_size // 2
        
        # 初始crop区域以mask中心为中心
        if attempt == 0:
            crop_left = max(0, center_x - half_size)
            crop_top = max(0, center_y - half_size)
        else:
            # 后续尝试添加一些随机偏移
            offset_x = random.randint(-half_size//2, half_size//2)
            offset_y = random.randint(-half_size//2, half_size//2)
            crop_left = max(0, min(img_w - image_size, center_x - half_size + offset_x))
            crop_top = max(0, min(img_h - image_size, center_y - half_size + offset_y))
        
        crop_right = min(img_w, crop_left + image_size)
        crop_bottom = min(img_h, crop_top + image_size)
        
        # 调整crop区域确保尺寸正确
        if crop_right - crop_left < image_size:
            crop_left = max(0, crop_right - image_size)
        if crop_bottom - crop_top < image_size:
            crop_top = max(0, crop_bottom - image_size)
        
        # 提取crop区域的mask
        cropped_mask_array = mask_array[crop_top:crop_bottom, crop_left:crop_right]
        
        # 计算有效区域比例
        total_pixels = cropped_mask_array.size
        valid_pixels = np.sum(cropped_mask_array > 0)
        valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0
        
        # 如果有效区域大于50%，接受这个crop
        if valid_ratio > 0.5:
            cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            cropped_mask = mask.crop((crop_left, crop_top, crop_right, crop_bottom))
            return cropped_img, cropped_mask
    
    # 如果多次尝试都没有找到合适的crop，返回包含最多mask区域的crop
    crop_left = max(0, center_x - half_size)
    crop_top = max(0, center_y - half_size)
    crop_right = min(img_w, crop_left + image_size)
    crop_bottom = min(img_h, crop_top + image_size)
    
    # 调整crop区域确保尺寸正确
    if crop_right - crop_left < image_size:
        crop_left = max(0, crop_right - image_size)
    if crop_bottom - crop_top < image_size:
        crop_top = max(0, crop_bottom - image_size)
    
    cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    cropped_mask = mask.crop((crop_left, crop_top, crop_right, crop_bottom))
    
    return cropped_img, cropped_mask

def load_sample_to_dict(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load a single sample from the dataset into a structured dictionary.
    
    Args:
        sample: Raw sample from webdataset
        
    Returns:
        Structured dictionary containing all the data
    """
    result = {
        'key': sample.get('__key__'),
        'url': sample.get('__url__'),
        'local_path': sample.get('__local_path__'),
        'image': None,
        'image_size': None,
        'mask': None
    }
    
    single_image = "jpg" in sample.keys()
    if single_image:
        result['image'] = sample['jpg']
        
        if result['image'] and hasattr(result['image'], 'size'):
            result['image_size'] = result['image'].size
    else:
        image_keys = [key for key in sample.keys() if key.endswith('jpg')]
        if image_keys:
            image_keys.sort()
            selected_image_key = random.choice(image_keys)
            selected_image_idx = int(selected_image_key[:-4])

            result['image'] = sample.get(selected_image_key)
            result['selected_image_idx'] = selected_image_idx
            
            if result['image'] and hasattr(result['image'], 'size'):
                result['image_size'] = result['image'].size

    
    # Process pickle data
    if 'pickle' in sample:
        pickle_data = sample['pickle']
        if isinstance(pickle_data, bytes):
            metadata_list = pickle.loads(pickle_data)
        else:
            metadata_list = pickle_data
        
        if isinstance(metadata_list, dict):
            metadata_list = [metadata_list]
            
        if metadata_list and len(metadata_list) > 0:
            selected_image_idx = result.get('selected_image_idx', 0)
                
            if selected_image_idx < len(metadata_list):
                metadata = metadata_list[selected_image_idx]
            else:
                metadata = metadata_list[0]                
            result.update({
                'img_id': metadata.get('img_id'),
                'ann_id': metadata.get('ann_id'),
                'image_path': metadata.get('image'),
                'category': metadata.get('category'),
                'caption': metadata.get('caption')
            })
            
            if isinstance(metadata['mask_rle'], list):
                mask_rle = metadata['mask_rle'][selected_image_idx]
            else:
                mask_rle = metadata['mask_rle']
            mask = decode_rle_mask(mask_rle)
            if isinstance(mask, np.ndarray):
                mask = mask * 255
                mask = Image.fromarray(mask, mode='L')
            result['mask'] = mask
            result['mask_size'] = mask.size

    result['image'] = short_side_resize(result['image'], 512, Image.Resampling.BICUBIC)
    result['mask'] = short_side_resize(result['mask'], 512, Image.Resampling.NEAREST)
    result['image'], result['mask'] = crop_image_and_mask(result['image'], result['mask'], 384)
        
    return result

from .VQVAE import get_vqvae_processor, VQVAE
def construct_messages(sample: Dict[str, Any], vae: VQVAE) -> Dict[str, Any]:
    """
    Construct messages from the sample.
    """
    # if "jpg" in sample and sample["jpg"] is None:
    # print(sample)
    results = load_sample_to_dict(sample)
    mask = results['mask']
    mask = mask.resize((128, 128), Image.Resampling.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    
    indices = vae.encode(mask).cpu().numpy().flatten().tolist()

    seg_token = ["<seg{:03d}>".format(i) for i in indices] # token id from 0-127
    seg_token = "".join(seg_token)
    seg_token = "<seg_begin>" + seg_token + "<seg_end>"
    default_prompt = "According the descritions, give the segmentation masks."
    messages = [
        {"role": "user", "content": default_prompt + "\n" + results['caption']},
        {"role": "assistant", "content": seg_token}
    ]
    images = results["image"]
    return dict(messages=messages, images=[images])

from functools import partial
def get_process_mask_func():
    vae = get_vqvae_processor()
    return partial(construct_messages, vae=vae)