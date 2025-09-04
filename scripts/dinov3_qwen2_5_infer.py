import os
import sys
import torch
import numpy as np
import glob
import random
import re
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
sys.path.append("/jfs/jensen/code/LLaMA-Factory")

from transformers import AutoProcessor
from src.llamafactory.model.modeling_dinotxt_qwen2_5_vl import DINOv3ViTQwen2_5_VLForConditionalGeneration
from src.llamafactory.data.data_utils import get_process_mask_func, get_vqvae_processor
from webdataset.compat import WebDataset
from qwen_vl_utils import process_vision_info

def blend_mask_with_image(image, mask, alpha=0.5, color=[255, 0, 0]):
    """
    Blend mask with image for visualization.
    
    Args:
        image: PIL Image object or torch tensor
        mask: torch tensor mask or numpy array
        alpha: transparency of the mask overlay (0.0 to 1.0)
        color: RGB color for the mask overlay [R, G, B]
        
    Returns:
        PIL Image with mask overlay
    """
    # Convert torch tensor to PIL if needed
    if torch.is_tensor(image):
        if image.dim() == 4:  # Batch dimension
            image = image[0]
        if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
            # Clamp values to [0, 1] range if they are in tensor format
            image = torch.clamp(image, 0, 1)
            image = TF.to_pil_image(image)
        else:
            image = TF.to_pil_image(image.unsqueeze(0))
    
    # Convert torch tensor mask to numpy
    if torch.is_tensor(mask):
        if mask.dim() == 4:  # Batch dimension
            mask = mask[0]
        if mask.dim() == 3:  # CHW format
            mask = mask.squeeze(0)
        mask = mask.detach().cpu().numpy()
    
    # Ensure mask values are in [0, 1] range
    if mask.max() > 1.0:
        mask = mask / 255.0
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Ensure both image and mask have same spatial dimensions
    # If image is grayscale, convert to RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
    
    # Ensure mask is the same size as image
    target_size = (img_array.shape[1], img_array.shape[0])  # (width, height) for PIL
    if mask.shape[:2] != img_array.shape[:2]:
        # Resize mask to match image size
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(target_size, Image.NEAREST)
        mask = np.array(mask_pil) / 255.0
    
    # Create colored mask overlay
    colored_mask = np.zeros_like(img_array)
    colored_mask[mask > 0.5] = color
    
    # Blend image with mask
    blended = img_array.copy().astype(np.float32)
    mask_indices = mask > 0.5
    blended[mask_indices] = (1 - alpha) * blended[mask_indices] + alpha * colored_mask[mask_indices]
    
    # Convert back to PIL Image
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def create_visualization_grid(images, masks, predictions, save_path, n_cols=4):
    """
    Create a grid visualization showing original images, ground truth masks, and predictions.
    
    Args:
        images: List of PIL Images or torch tensors
        masks: List of ground truth masks (torch tensors)
        predictions: List of predicted masks (torch tensors)
        save_path: Path to save the visualization
        n_cols: Number of columns in the grid
        
    Returns:
        PIL Image: The created grid visualization
    """
    n_samples = min(len(images), len(masks), len(predictions))
    if n_samples == 0:
        return None
    
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Create visualizations for each sample
    vis_images = []
    for i in range(n_samples):
        # Original image with GT mask overlay (red)
        gt_vis = blend_mask_with_image(images[i], masks[i], alpha=0.6, color=[255, 0, 0])
        
        # Original image with prediction overlay (green)
        pred_vis = blend_mask_with_image(images[i], predictions[i], alpha=0.6, color=[0, 255, 0])
        
        # Combine horizontally: [original + GT] | [original + pred]
        combined_width = gt_vis.width + pred_vis.width
        combined = Image.new('RGB', (combined_width, gt_vis.height))
        combined.paste(gt_vis, (0, 0))
        combined.paste(pred_vis, (gt_vis.width, 0))
        
        vis_images.append(combined)
    
    # Calculate grid dimensions
    if vis_images:
        img_width, img_height = vis_images[0].size
        grid_width = n_cols * img_width
        grid_height = n_rows * img_height
        grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        # Place each visualization in the grid
        for i, vis_image in enumerate(vis_images):
            row = i // n_cols
            col = i % n_cols
            x = col * img_width
            y = row * img_height
            grid_image.paste(vis_image, (x, y))
        
        # Save grid
        grid_image.save(save_path)
        print(f"Visualization saved to: {save_path}")
        
        return grid_image
    
    return None


def _find_tar_files(data_path: str):
    """Find all tar files in the data path."""
    if os.path.isdir(data_path):
        pattern = os.path.join(data_path, "**", "*.tar")
        tar_files = glob.glob(pattern, recursive=True)
        if not tar_files:
            raise FileNotFoundError(f"No tar files found in {data_path}")
        tar_files.sort()
        return tar_files
    else:
        return glob.glob(data_path)


def parse_mask_tokens(text):
    """Parse mask tokens from text content."""
    pattern = r"<seg_begin>(.*?)<seg_end>"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        return None
    
    # Get the first match and parse token IDs
    match = matches[0]
    token_ids = re.findall(r"<seg(\d+)>", match)
    token_ids = [int(tid) for tid in token_ids]
    
    # Limit to 64 tokens if necessary
    if len(token_ids) > 64:
        token_ids = token_ids[:64]
    
    return token_ids


def decode_mask_tokens(token_ids, vae):
    """Decode mask tokens to mask image using VAE."""
    if token_ids is None or len(token_ids) == 0:
        return None
    
    # Ensure we have exactly 64 tokens for 8x8 mask
    if len(token_ids) != 64:
        # Pad or truncate to 64 tokens
        if len(token_ids) < 64:
            token_ids = token_ids + [0] * (64 - len(token_ids))
        else:
            token_ids = token_ids[:64]
    
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).reshape(1, 8, 8)
    
    with torch.no_grad():
        mask = vae.decode(token_ids_tensor)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze(0).squeeze(0).sigmoid()
        mask = (mask + 1) / 2
        mask = torch.clamp(mask, 0, 1)
        return mask
    
    return None


def resize_mask_to_image(mask, target_size):
    """Resize mask to match image size."""
    if mask is None:
        return None
    
    # Convert tensor to PIL Image for resizing
    mask_pil = TF.to_pil_image(mask.unsqueeze(0))
    mask_resized = mask_pil.resize(target_size, Image.NEAREST)
    
    # Convert back to tensor
    mask_tensor = TF.to_tensor(mask_resized).squeeze(0)
    
    return mask_tensor


def add_text_to_image(image, text, position="top", font_size=20, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Add text label to image."""
    # Create a copy of the image
    img_with_text = image.copy()
    draw = ImageDraw.Draw(img_with_text)
    
    # Try to use a default font, fallback to basic font if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position
    img_width, img_height = image.size
    if position == "top":
        x = (img_width - text_width) // 2
        y = 10
    elif position == "bottom":
        x = (img_width - text_width) // 2
        y = img_height - text_height - 10
    else:
        x, y = position
    
    # Draw background rectangle
    padding = 5
    bg_bbox = [x - padding, y - padding, x + text_width + padding, y + text_height + padding]
    draw.rectangle(bg_bbox, fill=bg_color)
    
    # Draw text
    draw.text((x, y), text, font=font, fill=color)
    
    return img_with_text


def main():
    """Main function to run the mask prediction and visualization pipeline."""
    os.makedirs("./vis", exist_ok=True)
    # 1. Load data and initialize model
    print("Loading dataset and model...")
    data_path = "/jfs/qwen_models/describe-anything-dataset"
    data_files = _find_tar_files(data_path)
    random.shuffle(data_files)
    dataset_processor = get_process_mask_func()

    dataset = WebDataset(
        data_files, 
        shardshuffle=100,
    ).shuffle(100).decode("pil").map(dataset_processor)

    model_dir = "/data1/saves/qwen2_5vl-7b/full/debug"
    processor = AutoProcessor.from_pretrained(model_dir)
    model = DINOv3ViTQwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        attn_implementation="flash_attention_2"
    )
    vae = get_vqvae_processor()
    
    print("Processing samples...")
    for i, data in enumerate(dataset):
        print(f"\n--- Sample {i+1} ---")
        print("Data keys:", list(data.keys()))
        
        # Get original image and GT mask
        original_image = data["images"][0]  # PIL Image
        gt_mask_text = data["messages"][1]["content"]  # Ground truth mask tokens
        
        print(f"Image size: {original_image.size}")
        print(f"GT mask text preview: {gt_mask_text[:100]}...")
        
        # 2. Parse GT mask tokens
        gt_token_ids = parse_mask_tokens(gt_mask_text)
        if gt_token_ids is None:
            print("Failed to parse GT mask tokens, skipping...")
            continue
        print(f"GT token count: {len(gt_token_ids)}")
        
        # 3. Model inference
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": original_image},
                {"type": "text", "text": data["messages"][0]["content"]}
            ]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.5, do_sample=True)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"Model output: {output_text}")
        
        # 4. Parse predicted mask tokens
        pred_token_ids = parse_mask_tokens(output_text)
        if pred_token_ids is None:
            print("No mask tokens found in prediction, skipping...")
            continue
        print(f"Predicted token count: {len(pred_token_ids)}")
        
        # 5. Decode masks using VAE
        gt_mask = decode_mask_tokens(gt_token_ids, vae)
        pred_mask = decode_mask_tokens(pred_token_ids, vae)
        
        if gt_mask is None or pred_mask is None:
            print("Failed to decode masks, skipping...")
            continue
        
        # 6. Resize masks to image size and create visualizations
        target_size = original_image.size  # (width, height)
        gt_mask_resized = resize_mask_to_image(gt_mask, target_size)
        pred_mask_resized = resize_mask_to_image(pred_mask, target_size)
        
        # Create blended visualizations
        gt_vis = blend_mask_with_image(original_image, gt_mask_resized, alpha=0.6, color=[255, 0, 0])
        pred_vis = blend_mask_with_image(original_image, pred_mask_resized, alpha=0.6, color=[0, 255, 0])
        
        # Add text labels to visualizations
        gt_vis_labeled = add_text_to_image(gt_vis, "Ground Truth", position="top", font_size=24, color=(255, 255, 255), bg_color=(255, 0, 0))
        pred_vis_labeled = add_text_to_image(pred_vis, "Prediction", position="top", font_size=24, color=(255, 255, 255), bg_color=(0, 255, 0))
        
        # Save individual visualizations
        gt_vis_labeled.save(f"./vis/gt_mask_sample_{i+1}.png")
        pred_vis_labeled.save(f"./vis/pred_mask_sample_{i+1}.png")
        
        # Create comparison grid with labels
        combined_width = gt_vis_labeled.width + pred_vis_labeled.width
        combined = Image.new('RGB', (combined_width, gt_vis_labeled.height))
        combined.paste(gt_vis_labeled, (0, 0))
        combined.paste(pred_vis_labeled, (gt_vis_labeled.width, 0))
        combined.save(f"./vis/comparison_sample_{i+1}.png")
        
        print(f"Visualizations saved:")
        print(f"  - GT mask: ./vis/gt_mask_sample_{i+1}.png")
        print(f"  - Predicted mask: ./vis/pred_mask_sample_{i+1}.png")
        print(f"  - Comparison: ./vis/comparison_sample_{i+1}.png")
        
        # Process only first few samples for testing
        if i >= 10:
            break


if __name__ == "__main__":
    main()