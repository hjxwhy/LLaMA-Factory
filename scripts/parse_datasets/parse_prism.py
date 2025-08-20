# view_<i>/
#     rgb: RGB image as (H,W,3) array
#     xyz: Back-projected point-cloud (from RGB-D view) as (H,W,3) array of XYZ points
#     seg: Segmentation map as (H,W) array where each pixel is index of object name in object_names
#     object_names: List of object names visible in view
#     normals (optional): Point-cloud normals as (H,W,3) array
#     view_pose: Camera pose in world frame as (4,4) array
#     cam_params: Camera intrinsics matrix as (3,3) array

#     obs_<j>/
#         grasp_pose: Grasp pose in camera frame as (4,4) array
#         grasp_point: Point being grasped in camera frame as (3,) array
#         grasp_point_px: Point being grasped projected onto image plane as (2,) array
#         annot: YAML-formatted object with the following keys: ["annotation_id", "grasp_description", "object_description", "object_category", "object_id", "grasp_id"]

import os
import tarfile
import tempfile
import json
from pathlib import Path

import datasets
import huggingface_hub as hf_hub
import h5py
from PIL import Image
import numpy as np

from qwen_vl_utils import smart_resize

try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False
    print("webdataset not available, will use manual tar extraction")

def point_to_xml(grasp_pt: np.ndarray):
    if grasp_pt.ndim == 2:
        assert grasp_pt.shape == (1, 2)
        grasp_pt = grasp_pt[0]
    assert grasp_pt.shape == (2,)
    point_desc = "Where to grasp the object"
    return f"<point x=\"{grasp_pt[0]*100:.1f}\" y=\"{grasp_pt[1]*100:.1f}\" alt=\"{point_desc}\">{point_desc}</point>"

def map_sample(file_loc_map: dict[str, str], ex: dict):
    h5_path = file_loc_map[ex["scene_path"]]
    with h5py.File(h5_path, "r") as f:
        img = Image.fromarray(f[ex["view_id"]]["rgb"][:])
        grasp_pt_px = f[ex["view_id"]][ex["obs_id"]]["grasp_point_px"][:]
        grasp_pt_px = grasp_pt_px / np.array([img.width, img.height])
    task = ex["task"]
    prompt = f"Point to the grasp that would accomplish the following task: {task}"
    point_xml = point_to_xml(grasp_pt_px)
    response = f"In order to accomplish the task \"{task}\", the optimal grasp is described as follows: \"{ex['matching_grasp_desc']}\".\n\n{point_xml}"

    return dict(
        image=img,
        prompt=prompt,
        text=response,
        style="pointing",
        ex=ex["scene_id"] + "_" + ex["view_id"] + "_" + ex["obs_id"],
    )

def build_pointing_dataset(split: str, num_proc: int = 10) -> datasets.Dataset:
    # hf_fs = hf_hub.HfFileSystem()
    # chunks = hf_fs.ls(f"datasets/allenai/PRISM/PRISM-{split}", detail=False)
    # urls = []
    # for chunk in chunks:
    #     path = chunk[len("datasets/allenai/PRISM/"):]
    #     urls.append(hf_hub.hf_hub_url(repo_id="allenai/PRISM", filename=path, repo_type="dataset"))

    # dl_manager = datasets.DownloadManager(dataset_name="allenai/PRISM", record_checksums=False)
    # paths = dl_manager.download_and_extract(urls)

    file_loc_map = {}
    paths = [f"/DATA/disk0/data/PRISM/PRISM-{split}"]
    tar_files = []
    for path in paths:
        path = str(path)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            file_loc_map[file] = file_path
            # Collect all tar files for loading
            if file.endswith('.tar') and 'chunk_' in file:
                tar_files.append(file_path)
    
    # Sort tar files by chunk number to ensure consistent ordering
    tar_files.sort(key=lambda x: int(x.split('chunk_')[1].split('_of_')[0]))
    
    print(f"Found {len(tar_files)} tar files")
    
    # Load metadata first
    print("Loading metadata dataset...")
    csv_path = f"/DATA/disk0/data/PRISM/{split}.csv"
    metadata_ds = datasets.load_dataset("csv", data_files=csv_path, streaming=True)['train']
    
    # Create a generator that processes tar files one by one
    def process_tar_files_incrementally():
        current_file_map = {}
        current_temp_files = []
        
        # Group metadata by scene_path for efficient processing
        metadata_by_scene = {}
        print("Grouping metadata by scene...")
        for ex in metadata_ds:
            scene_path = ex["scene_path"]
            if scene_path not in metadata_by_scene:
                metadata_by_scene[scene_path] = []
            metadata_by_scene[scene_path].append(ex)
        
        print(f"Found metadata for {len(metadata_by_scene)} scenes")
        
        # Process each tar file
        for i, tar_path in enumerate(tar_files):
            print(f"Processing tar {i+1}/{len(tar_files)}: {os.path.basename(tar_path)}")
            
            # Clear previous temporary files
            for temp_file in current_temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            current_temp_files = []
            current_file_map = {}
            
            if WEBDATASET_AVAILABLE:
                # Process current tar file with webdataset
                dataset = wds.WebDataset([tar_path], shardshuffle=False).decode()
                
                for sample in dataset:
                    if 'hdf5' in sample:
                        filename = sample['__key__'] + '.hdf5'
                        
                        # Create temporary file for the HDF5 content
                        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp_file:
                            tmp_file.write(sample['hdf5'])
                            tmp_file.flush()
                            current_file_map[filename] = tmp_file.name
                            current_temp_files.append(tmp_file.name)
                        
                        # Process all metadata entries for this scene
                        if filename in metadata_by_scene:
                            for metadata_entry in metadata_by_scene[filename]:
                                try:
                                    result = map_sample(current_file_map, metadata_entry)
                                    yield result
                                except Exception as e:
                                    print(f"Error processing {filename}: {e}")
                                    continue
            else:
                # Extract tar file temporarily
                with tempfile.TemporaryDirectory() as temp_dir:
                    with tarfile.open(tar_path, 'r') as tar:
                        tar.extractall(temp_dir)
                        
                        for member in tar.getmembers():
                            if member.isfile() and member.name.endswith('.hdf5'):
                                extracted_path = os.path.join(temp_dir, member.name)
                                filename = member.name
                                current_file_map[filename] = extracted_path
                                
                                # Process all metadata entries for this scene
                                if filename in metadata_by_scene:
                                    for metadata_entry in metadata_by_scene[filename]:
                                        try:
                                            result = map_sample(current_file_map, metadata_entry)
                                            yield result
                                        except Exception as e:
                                            print(f"Error processing {filename}: {e}")
                                            continue
        
        # Clean up remaining temporary files
        for temp_file in current_temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    # Create dataset from generator
    final_dataset = datasets.IterableDataset.from_generator(process_tar_files_incrementally)
    
    return final_dataset

if __name__ == "__main__":
    split = "test"
    dataset = build_pointing_dataset(split)
    save_path = f"/DATA/disk1/data/PRISM/{split}"
    os.makedirs(save_path, exist_ok=True)
    conversations = []
    cache_saved_path = {}
    for i, data in enumerate(dataset):
        # {'image': <PIL.Image.Image image mode=RGB size=640x480 at 0x7FE662C6CA50>, 'prompt': 'Point to the grasp that would accomplish the following task: Tilt the milk carton to allow pouring its contents out.', 'text': 'In order to accomplish the task "Tilt the milk carton to allow pouring its contents out.", the optimal grasp is described as follows: "The grasp is on the side of the milk carton, near the middle. The fingers are pinching the carton from opposite sides, gripping the flat surfaces of the carton.".\n\n<point x="56.0" y="41.4" alt="Where to grasp the object">Where to grasp the object</point>', 'style': 'pointing'}
        img = data["image"]
        # x="56.0" y="41.4"
        w, h = img.size
        resized_h, resized_w = smart_resize(h, w)
        img = img.resize((resized_w, resized_h))
        x = float(data["text"].split("x=")[1].split(" ")[0].replace("\"", ""))
        y = float(data["text"].split("y=")[1].split(" ")[0].replace("\"", ""))

        # Update the text with new coordinates
        new_x = int(x / 100 * resized_w)
        new_y = int(y / 100 * resized_h)
        updated_text = data["text"].replace(f'x="{x}"', f'x="{new_x}"')
        updated_text = updated_text.replace(f'y="{y}"', f'y="{new_y}"')
        data["text"] = updated_text
        
        img_path = os.path.join(save_path, f"{i:06d}.jpg")
        ex = data["ex"]
        if ex not in cache_saved_path:
            try:
                img.save(img_path)
                cache_saved_path[ex] = img_path
            except Exception as e:
                print(f"Error saving image {i:06d}: {e}")
                continue
        else:
            img_path = cache_saved_path[ex]


        messages = [
            {"content": "<image>" + data["prompt"], "role": "user"},
            {"content": data["text"], "role": "assistant"}
        ]
        item = {
            "messages": messages,
            "images": [img_path.replace("/DATA/disk1/data/", "")],
            "id": f"prism_pointing_{i:06d}"
        }
        conversations.append(item)


    with open(os.path.join(os.path.dirname(save_path), f"conversations_{split}.json"), "w") as f:
        json.dump(conversations, f, indent=2)
        # breakpoint()
        # x = float(data["text"].split("x=")[1].split(" ")[0].replace("\"", ""))
        # y = float(data["text"].split("y=")[1].split(" ")[0].replace("\"", ""))
        # x = x / 100 * w
        # y = y / 100 * h
        # # Draw the point on the image
        # from PIL import ImageDraw
        # draw = ImageDraw.Draw(img)
        # point_size = 5
        # draw.ellipse([new_x-point_size, new_y-point_size, new_x+point_size, new_y+point_size], fill='red', outline='red')

        # img.save("test.png")
        # exit()

    # build_pointing_dataset("test")
