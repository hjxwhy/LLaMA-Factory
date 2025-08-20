import os
import json
import zipfile
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from tqdm import tqdm
from datasets import load_dataset
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_zip_file(zip_path, extract_to):
    """
    Extract a single zip file to the specified directory, preserving folder structure
    """
    try:
        zip_filename = os.path.basename(zip_path)
        logger.info(f"Starting extraction of {zip_filename}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Create extraction directory if it doesn't exist
            os.makedirs(extract_to, exist_ok=True)
            
            # Extract all files preserving directory structure
            zip_ref.extractall(extract_to)
            
        logger.info(f"Successfully extracted {zip_filename}")
        return f"Success: {zip_filename}"
        
    except Exception as e:
        error_msg = f"Error extracting {zip_filename}: {str(e)}"
        logger.error(error_msg)
        return error_msg

def parallel_extract_zip_files(zip_files, data_root, unzip_root, max_workers=None):
    """
    Extract multiple zip files in parallel
    """
    if max_workers is None:
        # Use fewer workers to avoid overwhelming the system
        max_workers = min(len(zip_files), multiprocessing.cpu_count() // 2)
    
    logger.info(f"Starting parallel extraction of {len(zip_files)} zip files using {max_workers} workers")
    
    # Prepare full paths for zip files
    zip_paths = [os.path.join(data_root, zip_file) for zip_file in zip_files]
    
    # Use ProcessPoolExecutor for parallel extraction
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all extraction tasks
        future_to_zip = {
            executor.submit(extract_zip_file, zip_path, unzip_root): zip_path 
            for zip_path in zip_paths
        }
        
        # Process completed tasks
        results = []
        for future in as_completed(future_to_zip):
            zip_path = future_to_zip[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                error_msg = f"Exception occurred for {zip_path}: {exc}"
                logger.error(error_msg)
                results.append(error_msg)
    
    return results

def copy_json_file(source_file, target_dir):
    """
    Copy JSON annotation file to target directory
    """
    try:
        if not os.path.exists(source_file):
            logger.warning(f"JSON file not found: {source_file}")
            return False
            
        os.makedirs(target_dir, exist_ok=True)
        target_file = os.path.join(target_dir, os.path.basename(source_file))
        
        shutil.copy2(source_file, target_file)
        logger.info(f"Successfully copied {os.path.basename(source_file)} to {target_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error copying JSON file: {str(e)}")
        return False

def extract_m4_instruct_data(data_root, unzip_root, max_workers=None):
    """
    Extract M4-Instruct dataset zip files and copy annotation file
    
    Args:
        data_root (str): Source directory containing zip files and annotations
        unzip_root (str): Target directory for extraction
        max_workers (int, optional): Number of parallel workers. Defaults to CPU_count//2
    
    Returns:
        dict: Summary of extraction results
    """
    logger.info("=" * 60)
    logger.info("Starting M4-Instruct Data Extraction")
    logger.info("=" * 60)
    
    # Get list of zip files
    if not os.path.exists(data_root):
        logger.error(f"Source directory not found: {data_root}")
        return {"success": False, "error": "Source directory not found"}
    
    zip_files = os.listdir(data_root)
    zip_files = [f for f in zip_files if f.endswith(".zip")]
    
    if not zip_files:
        logger.warning("No zip files found to extract")
        return {"success": False, "error": "No zip files found"}
    
    # Extract zip files in parallel
    logger.info(f"Found {len(zip_files)} zip files to extract")
    logger.info(f"Extracting from: {data_root}")
    logger.info(f"Extracting to: {unzip_root}")
    
    results = parallel_extract_zip_files(zip_files, data_root, unzip_root, max_workers)
    
    # Analyze results
    successful = [r for r in results if r.startswith("Success")]
    failed = [r for r in results if not r.startswith("Success")]
    
    logger.info(f"Zip extraction completed: {len(successful)} successful, {len(failed)} failed")
    
    if failed:
        logger.error("Failed extractions:")
        for failure in failed:
            logger.error(failure)
    
    # Copy JSON annotation file
    m4_instruct_ann = os.path.join(data_root, "m4_instruct_annotations.json")
    json_copied = copy_json_file(m4_instruct_ann, unzip_root)
    
    # Prepare summary
    summary = {
        "success": True,
        "total_zip_files": len(zip_files),
        "successful_extractions": len(successful),
        "failed_extractions": len(failed),
        "json_file_copied": json_copied,
        "failed_files": [f.split(": ", 1)[1] if ": " in f else f for f in failed],
        "extract_location": unzip_root
    }
    
    logger.info("=" * 60)
    logger.info("Extraction Summary:")
    logger.info(f"  Total ZIP files: {summary['total_zip_files']}")
    logger.info(f"  Successful: {summary['successful_extractions']}")
    logger.info(f"  Failed: {summary['failed_extractions']}")
    logger.info(f"  JSON file copied: {summary['json_file_copied']}")
    logger.info(f"  Target location: {summary['extract_location']}")
    logger.info("=" * 60)
    
    return summary

# Main execution
if __name__ == "__main__":
    data_root = "/localfolder/data/M4-Instruct-Data"
    unzip_root = "/DATA/disk1/data/M4-Instruct-Data"
    
    # Extract M4-Instruct data
    # result = extract_m4_instruct_data(data_root, unzip_root)
    
    # if result["success"]:
    #     logger.info("M4-Instruct data extraction completed successfully!")
    # else:
    #     logger.error(f"M4-Instruct data extraction failed: {result.get('error', 'Unknown error')}")

    # /localfolder/data/LLaVA-NeXT-Data
    data_root = "/localfolder/data/LLaVA-NeXT-Data"
    unzip_root = "/DATA/disk1/data/LLaVA-NeXT-Data/images"

    os.makedirs(unzip_root, exist_ok=True)

    # Extract LLaVA-NeXT data
    # result = extract_llava_next_data(data_root, unzip_root)
    dataset = load_dataset(data_root, streaming=True)["train"]
    conversations = []
    for i, item in enumerate(dataset):
        image = item.pop("image")
        if image is None:
            continue
        try:
            image_path = os.path.join(unzip_root, f"{i:06d}.png")
            image = image.convert("RGB")
            image.save(image_path)
            item["image"] = [image_path.replace("/DATA/disk1/data/", "")]
            # remove "Answer the question with GPT-T-COCO format." in values
            for conv in item["conversations"]:
                if conv["from"] == "human":
                    conv["value"] = conv["value"].replace("Answer the question with GPT-T-COCO format.", "")
                    # revemo "\n" in the end string
                    conv["value"] = conv["value"].rstrip("\n")
        except Exception as e:
            logger.error(f"Error saving image {i}: {e}")
            continue
        conversations.append(item)
    with open(os.path.join(os.path.dirname(unzip_root), "llava_next_conversations.json"), "w") as f:
        json.dump(conversations, f, indent=2)
