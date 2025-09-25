import tensorflow_datasets as tfds
import tyro
import os
from tqdm import tqdm
import numpy as np
import random
from PIL import Image
import tensorflow as tf
import glob

def process_episode(episode, camera_views, output_dir, index):
    """处理单个 episode，从 steps Dataset 中随机抽取一个 step 并保存图像"""
    steps = episode['steps']

    # 检查是否有数据（尝试获取一个元素看是否为空）
    try:
        _ = steps.take(1).get_single_element()
    except tf.errors.InvalidArgumentError:
        return None  # 空数据集

    # 随机抽取一个 step：先打乱，取第一个
    # buffer_size 应该 >= 总 step 数，但为效率可设为较小值（如 1000）
    # 如果 steps 很多，可设 buffer_size=1000；如果少，设 buffer_size=大数
    try:
        selected_step = steps.shuffle(buffer_size=1000).take(1).get_single_element()
    except Exception:
        return None

    # 获取指令（如果存在）
    instruction = ""
    if 'language_instruction' in selected_step:
        instruction = selected_step['language_instruction'].numpy().decode('utf-8')

    saved_paths = []
    for camera_view in camera_views:
        try:
            image_tensor = selected_step['observation'][camera_view]  # 现在 selected_step 是 dict
            image_array = image_tensor.numpy()

            if image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)

            if image_array.shape != (224, 224, 3):
                continue

            image_pil = Image.fromarray(image_array, 'RGB')
            camera_dir = os.path.join(output_dir, camera_view)
            image_path = os.path.join(camera_dir, f"{index:06d}.jpg")
            image_pil.save(image_path, 'JPEG', quality=95)
            saved_paths.append(image_path)

        except Exception:
            continue

    return instruction, saved_paths

def discover_datasets(data_dir: str):
    """自动发现数据目录中的所有可用数据集"""
    dataset_pattern = os.path.join(data_dir, "*", "1.0.0")
    dataset_dirs = glob.glob(dataset_pattern)
    dataset_names = []
    
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(os.path.dirname(dataset_dir))
        dataset_names.append(dataset_name)
    
    return sorted(dataset_names)

def process_single_dataset(dataset_name: str, data_dir: str, output_dir: str, global_index_start: int = 0):
    """处理单个数据集"""
    print(f"\n🔄 Processing dataset: {dataset_name}")
    print("=" * 50)
    
    # 加载数据集
    try:
        ds = tfds.load(dataset_name, split='train', data_dir=data_dir, shuffle_files=True)
        print(f"✅ Successfully loaded dataset: {dataset_name}")
    except Exception as e:
        print(f"❌ Failed to load dataset {dataset_name}: {e}")
        return global_index_start, []

    # 创建输出目录
    camera_views = ['image_camera_head', 'image_camera_wrist_left', 'image_camera_wrist_right']
    for camera_view in camera_views:
        os.makedirs(os.path.join(output_dir, camera_view), exist_ok=True)

    # 优化数据管道：预取 + 并行处理
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # 使用 tqdm 手动控制进度（避免嵌套在 tf.data 里）
    try:
        total = ds.cardinality().numpy()
        if total == tf.data.UNKNOWN_CARDINALITY:
            total = None  # 未知大小
    except:
        total = None

    print(f"Total episodes: {total if total else 'unknown'}")

    instructions = []
    current_index = global_index_start

    # 遍历数据集（不使用 .map() 因为要控制文件名索引）
    for i, episode in enumerate(tqdm(ds, total=total, desc=f"Processing {dataset_name}")):
        result = process_episode(episode, camera_views, output_dir, current_index)
        if result is None:
            continue
        instruction, saved_paths = result
        if instruction:
            instructions.append(f"Episode {current_index:06d} ({dataset_name}): {instruction}")
        current_index += 1

    # 打印该数据集的指令（避免频繁 I/O）
    if instructions:
        print(f"\n=== Language Instructions for {dataset_name} ===")
        for inst in instructions[:5]:  # 只打印前5条示例
            print(inst)
        if len(instructions) > 5:
            print(f"... and {len(instructions) - 5} more.")

    print(f"✅ Completed {dataset_name}: processed {current_index - global_index_start} episodes")
    return current_index, instructions

def main(
    data_dir: str,
    output_dir: str = "extracted_images",
    dataset_name: str = None,  # 如果指定，只处理这个数据集
    num_parallel_calls: int = tf.data.AUTOTUNE,  # 自动并行
    batch_size: int = 32,  # 批量处理
):
    print("🚀 Starting Galaxea dataset processing...")
    print("=" * 60)
    
    # 创建主输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果指定了特定数据集，只处理那个数据集
    if dataset_name:
        print(f"📁 Processing single dataset: {dataset_name}")
        datasets_to_process = [dataset_name]
    else:
        # 自动发现所有可用数据集
        print("🔍 Discovering available datasets...")
        datasets_to_process = discover_datasets(data_dir)
        
        if not datasets_to_process:
            print("❌ No datasets found in the specified directory!")
            return
        
        print(f"📁 Found {len(datasets_to_process)} datasets:")
        for i, ds_name in enumerate(datasets_to_process, 1):
            print(f"  {i}. {ds_name}")
        print()
    
    # 处理所有数据集
    all_instructions = []
    global_index = 0
    total_episodes = 0
    
    for i, ds_name in enumerate(datasets_to_process, 1):
        print(f"\n📊 Progress: {i}/{len(datasets_to_process)} datasets")
        
        # 处理单个数据集
        new_global_index, dataset_instructions = process_single_dataset(
            ds_name, data_dir, output_dir, global_index
        )
        
        episodes_processed = new_global_index - global_index
        total_episodes += episodes_processed
        global_index = new_global_index
        all_instructions.extend(dataset_instructions)
        
        print(f"📈 Running total: {total_episodes} episodes processed so far")
    
    # 最终统计
    print("\n" + "=" * 60)
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📊 Total datasets processed: {len(datasets_to_process)}")
    print(f"📊 Total episodes processed: {total_episodes}")
    print(f"📁 Images saved to: {output_dir}")
    
    # 显示目录结构
    print(f"\n📁 Directory structure:")
    for camera_view in ['image_camera_head', 'image_camera_wrist_left', 'image_camera_wrist_right']:
        camera_dir = os.path.join(output_dir, camera_view)
        if os.path.exists(camera_dir):
            image_count = len([f for f in os.listdir(camera_dir) if f.endswith('.jpg')])
            print(f"  {camera_view}/: {image_count} images")
    
    # 显示一些指令示例
    if all_instructions:
        print(f"\n📝 Language Instructions (showing 10 examples):")
        for inst in all_instructions[:10]:
            print(f"  {inst}")
        if len(all_instructions) > 10:
            print(f"  ... and {len(all_instructions) - 10} more instructions.")
    
    print(f"\n✅ All done! Check the {output_dir} directory for your extracted images.")


if __name__ == "__main__":
    tyro.cli(main)