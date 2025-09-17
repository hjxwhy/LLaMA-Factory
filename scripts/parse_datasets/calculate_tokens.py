import os
import json
import pandas as pd
from collections import defaultdict
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from decord import VideoReader
import numpy as np
from pathlib import Path
import re
from datasets import load_dataset
import multiprocessing as mp
from multiprocessing import Pool, Manager
import time
from tqdm import tqdm
import argparse

dataset_path = "data/data/vlm_mix_robot_openx_training_v5"
qwen_path = "/jfs/qwen_models/Qwen2.5-VL-7B-Instruct"
data_root = "/jfs/jensen/code/LLaMA-Factory/data/data"

# 并行处理配置
MAX_PROCESSES = 8  # 最大进程数
MIN_CHUNK_SIZE = 1000  # 最小chunk大小
CHUNK_MULTIPLIER = 4  # chunk数量是进程数的倍数

# 这个数据集有不同的数据类型，需要按类统计多少token，用Qwen2.5-VL preprocessor处理出token

# 判断是纯文本数据的方法： len(data["images"])==0 or data["images"] is None

# 其他数据集需要根据images路径判断属于哪个数据集

# - describe-anything-dataset  
# - EmbodiedScan  
# - FSD-Dataset  
# - LLaVA-NeXT-Data  
# - LLaVA-OneVision-Data-Parsed  
# - LLaVA-ReCap-558K-wds  
# - M4-Instruct-Data  
# - open-x  
# - pixmo-ask-model-anything-parse  
# - pixmo-points  
# - PRISM  
# - ShareRobot
# - tulu-3-sft-olmo-2-mixture-0225

# 需求:
# 1. 需要用Qwen2.5-VL preprocessor处理出token, 文本有若干个<image>占位符,用qwen preprocessor的时候需要把这些处理掉,preprocessor会自动处理图像
# 2. open-x的数据有一些需要特殊处理,特殊处理的参考代码如下:
#     if "open-x" in media_path:
#         media_path = media_path.replace("train_v3", "train")
#         # if "berkeley_autolab_ur5" in media_path:
#         #     if not os.path.exists(media_path):
#         #         media_path = media_path.replace(".jpg", ".png")
#         data_names = ["berkeley_autolab_ur5", "bridge", "fractal20220817_data", "jaco_play", "libero"]
#         if any(data_name in media_path for data_name in data_names):
#             pass
#         else:
#             media_path = media_path.replace(".jpg", ".mp4")
# 视频数据需要这样读取:
#     video_path = os.path.join(os.path.dirname(image) + ".mp4")
#     image_id = int(os.path.basename(image).split(".")[0])
#     video_reader = VideoReader(video_path)
#     video_frames = video_reader.get_batch([image_id])
#     image = Image.fromarray(video_frames.asnumpy()[0])

# 3. 代码需要有一些容灾的功能,有写图像会有问题会报错,但是也别写太多try-except,关键读图像的地方写一下就好
# 4. 需要把结果整理到一个markdown里,需要有每种数据的token数量和数据条数

def load_qwen_processor():
    """加载Qwen2.5-VL处理器"""
    processor = Qwen2VLProcessor.from_pretrained(qwen_path)
    return processor

def classify_dataset_type(images_path_list):
    """根据图像路径判断数据集类型"""
    if not images_path_list or len(images_path_list) == 0:
        return "text_only"
    
    # 检查第一个图像路径来判断数据集类型
    first_image_path = images_path_list[0] if images_path_list else ""
    
    dataset_types = [
        "describe-anything-dataset",
        "EmbodiedScan", 
        "FSD-Dataset",
        "LLaVA-NeXT-Data",
        "LLaVA-OneVision-Data-Parsed",
        "LLaVA-ReCap-558K-wds",
        "M4-Instruct-Data",
        "open-x",
        "pixmo-ask-model-anything-parse",
        "pixmo-points",
        "PRISM",
        "ShareRobot",
        "tulu-3-sft-olmo-2-mixture-0225"
    ]
    
    for dataset_type in dataset_types:
        if dataset_type in first_image_path:
            return dataset_type
    
    return "unknown"

def process_openx_image_path(media_path):
    """处理open-x数据的特殊路径转换"""
    # 确保media_path是字符串
    if not isinstance(media_path, str):
        return media_path
        
    if "open-x" not in media_path:
        return media_path
        
    media_path = media_path.replace("train_v3", "train")
    data_names = ["berkeley_autolab_ur5", "bridge", "fractal20220817_data", "jaco_play", "libero"]
    
    if any(data_name in media_path for data_name in data_names):
        return media_path
    else:
        return media_path.replace(".jpg", ".mp4")

def load_image_safely(image_path, is_video=False):
    """安全地加载图像，包含容灾功能"""
    try:
        if is_video:
            # 处理视频数据 - 修复路径构造
            # 从 xxx/0377.mp4 构造为 xxx.mp4
            if image_path.endswith('.mp4'):
                # 移除文件名末尾的数字部分
                path_parts = image_path.split('/')
                video_file = path_parts[-1]  # 如 0377.mp4
                video_dir = '/'.join(path_parts[:-1])  # 目录部分
                
                # 构造视频文件路径 - 移除帧号目录
                video_path_parts = video_dir.split('/')
                if video_path_parts[-1].isdigit():  # 最后一部分是数字（帧号目录）
                    video_base = '/'.join(video_path_parts[:-1])
                    video_path = os.path.join(data_root, video_base + ".mp4")
                    
                    # 提取帧号
                    frame_id = int(video_file.split('.')[0])
                else:
                    video_path = os.path.join(data_root, image_path)
                    frame_id = 0
                
                # 检查视频文件是否存在
                if not os.path.exists(video_path):
                    return None
                    
                video_reader = VideoReader(video_path)
                video_frames = video_reader.get_batch([frame_id])
                image = Image.fromarray(video_frames.asnumpy()[0])
                return image
            else:
                return None
        else:
            # 处理普通图像
            full_path = os.path.join(data_root, image_path)
            if os.path.exists(full_path):
                image = Image.open(full_path)
                return image
            else:
                return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def remove_image_placeholders(text):
    """移除文本中的<image>占位符"""
    if isinstance(text, str):
        return re.sub(r'<image>', '', text)
    else:
        return text  # 如果不是字符串，直接返回

def calculate_tokens_for_sample(processor, messages, images=None):
    """为单个样本计算token数量"""
    try:
        # 处理消息文本，移除<image>占位符，并确保格式正确
        processed_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                processed_msg = msg.copy()
                # 确保content字段是字符串
                if 'content' in processed_msg:
                    processed_msg['content'] = remove_image_placeholders(processed_msg['content'])
                processed_messages.append(processed_msg)
            else:
                # 如果不是dict，尝试直接处理
                processed_messages.append(msg)
        
        # 处理图像
        processed_images = []
        if images:
            for img_path in images:
                # 确保img_path是字符串
                if isinstance(img_path, str):
                    processed_path = process_openx_image_path(img_path)
                    is_video = processed_path.endswith('.mp4')
                    image = load_image_safely(processed_path, is_video)
                    if image is not None:
                        processed_images.append(image)
        
        # 构建文本用于processor - 转换为字符串格式
        text_content = ""
        for msg in processed_messages:
            if isinstance(msg, dict):
                role = msg.get('role', msg.get('from', 'user'))
                content = msg.get('content', msg.get('value', ''))
                text_content += f"{role}: {content}\n"
            else:
                text_content += str(msg) + "\n"
        
        # 使用processor计算token
        if processed_images:
            # 有图像的情况
            inputs = processor(text=text_content, images=processed_images, return_tensors="pt")
        else:
            # 纯文本的情况  
            inputs = processor(text=text_content, return_tensors="pt")
        
        # 计算token数量
        input_ids = inputs['input_ids']
        token_count = input_ids.shape[1]
        
        return token_count
        
    except Exception as e:
        print(f"Error calculating tokens: {e}")
        return 0

def debug_sample_format(dataset, num_samples=3):
    """调试函数：检查数据格式"""
    print("\n=== Debugging Sample Format ===")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"Keys: {list(sample.keys())}")
        
        if 'messages' in sample:
            print(f"Messages type: {type(sample['messages'])}")
            print(f"Messages length: {len(sample['messages']) if sample['messages'] else 0}")
            if sample['messages']:
                print(f"First message type: {type(sample['messages'][0])}")
                print(f"First message: {sample['messages'][0]}")
                if len(sample['messages']) > 1:
                    print(f"Second message: {sample['messages'][1]}")
        
        if 'images' in sample:
            print(f"Images type: {type(sample['images'])}")
            print(f"Images length: {len(sample['images']) if sample['images'] else 0}")
            if sample['images']:
                print(f"First image type: {type(sample['images'][0])}")
                print(f"First image path: {sample['images'][0]}")
        
        # 分类这个样本
        images = sample.get('images', [])
        dataset_type = classify_dataset_type(images)
        print(f"Classified as: {dataset_type}")
        
    print("=== End Debug ===\n")

def process_chunk(args):
    """处理数据块的worker函数"""
    start_idx, end_idx, chunk_id = args
    
    # 每个进程独立加载数据集和processor
    print(f"Process {chunk_id}: Loading dataset and processor...")
    dataset = load_dataset(dataset_path, split="train")
    processor = load_qwen_processor()
    
    # 本chunk的统计结果
    chunk_stats = defaultdict(lambda: {"token_count": 0, "sample_count": 0, "error_count": 0})
    
    processed_count = 0
    chunk_size = end_idx - start_idx
    
    for idx in range(start_idx, end_idx):
        try:
            sample = dataset[idx]
            
            # 解析数据 - 使用messages和images字段
            messages = sample.get('messages', [])
            images = sample.get('images', [])
            
            # 判断数据集类型
            dataset_type = classify_dataset_type(images)
            
            # 计算token数量
            token_count = calculate_tokens_for_sample(processor, messages, images)
            
            # 更新统计
            chunk_stats[dataset_type]["token_count"] += token_count
            chunk_stats[dataset_type]["sample_count"] += 1
            
            processed_count += 1
            
            # 每处理1000个样本报告进度
            if processed_count % 1000 == 0:
                print(f"Process {chunk_id}: Processed {processed_count}/{chunk_size} samples...")
                
        except Exception as e:
            print(f"Process {chunk_id}, Sample {idx}: Error - {e}")
            dataset_type = "error"
            chunk_stats[dataset_type]["error_count"] += 1
    
    print(f"Process {chunk_id}: Completed processing {processed_count} samples")
    return dict(chunk_stats)

def merge_stats(all_stats):
    """合并多个进程的统计结果"""
    merged_stats = defaultdict(lambda: {"token_count": 0, "sample_count": 0, "error_count": 0})
    
    for stats in all_stats:
        for dataset_type, data in stats.items():
            merged_stats[dataset_type]["token_count"] += data["token_count"]
            merged_stats[dataset_type]["sample_count"] += data["sample_count"]
            merged_stats[dataset_type]["error_count"] += data["error_count"]
    
    return dict(merged_stats)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Calculate tokens for VLM dataset")
    parser.add_argument("--processes", type=int, default=None, 
                       help=f"Number of processes (default: min(CPU_count, {MAX_PROCESSES}))")
    parser.add_argument("--chunk-size", type=int, default=None,
                       help=f"Chunk size per process (default: auto-calculated)")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode with limited samples")
    parser.add_argument("--debug-samples", type=int, default=100,
                       help="Number of samples to process in debug mode")
    return parser.parse_args()

def main():
    """主函数：处理数据集并统计token"""
    args = parse_args()
    start_time = time.time()
    
    # 使用datasets库加载数据
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        dataset = load_dataset(dataset_path, split="train")
        print(f"Loaded dataset with {len(dataset)} samples")
        
        # 调试数据格式（只看前几个样本）
        debug_sample_format(dataset, num_samples=1)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    total_samples = len(dataset)
    
    # Debug模式限制样本数量
    if args.debug:
        total_samples = min(total_samples, args.debug_samples)
        print(f"Debug mode: Processing only {total_samples} samples")
    
    # 配置并行处理参数
    num_processes = args.processes or min(mp.cpu_count(), MAX_PROCESSES)
    chunk_size = args.chunk_size or max(MIN_CHUNK_SIZE, total_samples // (num_processes * CHUNK_MULTIPLIER))
    
    print(f"Using {num_processes} processes with chunk size {chunk_size}")
    print(f"Estimated {(total_samples + chunk_size - 1) // chunk_size} chunks")
    
    # 将数据分成chunks（只传递索引范围）
    chunks = []
    for i in range(0, total_samples, chunk_size):
        end_idx = min(i + chunk_size, total_samples)
        chunks.append((i, end_idx, len(chunks)))
    
    print(f"Created {len(chunks)} chunks for processing")
    
    # 并行处理
    print("Starting parallel processing...")
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    # 合并结果
    print("Merging results...")
    dataset_stats = merge_stats(results)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nProcessing complete!")
    print(f"Total time: {processing_time:.2f} seconds")
    print(f"Average speed: {total_samples / processing_time:.2f} samples/second")
    
    # 生成markdown报告
    generate_markdown_report(dataset_stats, processing_time)
    
    return dataset_stats

def generate_markdown_report(dataset_stats, processing_time=None):
    """生成markdown格式的统计报告"""
    report_path = "token_analysis_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# VLM Mix Robot OpenX Training v5 Token Analysis Report\n\n")
        f.write(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if processing_time:
            f.write(f"Processing time: {processing_time:.2f} seconds\n")
        f.write("\n")
        
        f.write("## Summary\n\n")
        f.write("| Dataset Type | Sample Count | Total Tokens | Avg Tokens/Sample | Error Count |\n")
        f.write("|-------------|-------------|-------------|------------------|-------------|\n")
        
        total_samples = 0
        total_tokens = 0
        total_errors = 0
        
        # 按样本数量排序
        sorted_stats = sorted(dataset_stats.items(), key=lambda x: x[1]["sample_count"], reverse=True)
        
        for dataset_type, stats in sorted_stats:
            sample_count = stats["sample_count"]
            token_count = stats["token_count"]
            error_count = stats["error_count"]
            avg_tokens = token_count / sample_count if sample_count > 0 else 0
            
            total_samples += sample_count
            total_tokens += token_count
            total_errors += error_count
            
            f.write(f"| {dataset_type} | {sample_count:,} | {token_count:,} | {avg_tokens:.1f} | {error_count} |\n")
        
        f.write(f"| **Total** | **{total_samples:,}** | **{total_tokens:,}** | **{total_tokens/total_samples if total_samples > 0 else 0:.1f}** | **{total_errors}** |\n\n")
        
        f.write("## Detailed Statistics\n\n")
        
        for dataset_type, stats in sorted_stats:
            if stats["sample_count"] > 0:
                f.write(f"### {dataset_type}\n")
                f.write(f"- **Sample Count**: {stats['sample_count']:,}\n")
                f.write(f"- **Total Tokens**: {stats['token_count']:,}\n")
                f.write(f"- **Average Tokens per Sample**: {stats['token_count'] / stats['sample_count']:.1f}\n")
                f.write(f"- **Percentage of Total Samples**: {stats['sample_count'] / total_samples * 100:.1f}%\n")
                f.write(f"- **Percentage of Total Tokens**: {stats['token_count'] / total_tokens * 100:.1f}%\n")
                if stats["error_count"] > 0:
                    f.write(f"- **Error Count**: {stats['error_count']}\n")
                f.write("\n")
        
        f.write("## Notes\n\n")
        f.write("- Token counts are calculated using Qwen2.5-VL-7B-Instruct processor\n")
        f.write("- Text-only data is identified when images list is empty or None\n")
        f.write("- Dataset type is determined by the first image path in the images list\n")
        f.write("- Open-X data receives special path processing for video files\n")
        f.write("- Error samples are those that failed during processing\n")
    
    print(f"\nMarkdown report saved to: {report_path}")

if __name__ == "__main__":
    main()