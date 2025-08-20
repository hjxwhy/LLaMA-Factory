from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image, ImageDraw
import requests
import torch
import re
import json
import math
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # load the processor
    processor = AutoProcessor.from_pretrained(
        '/localfolder/data/Molmo-72B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    logger.info("Processor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load processor: {e}")
    raise

try:
    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        '/localfolder/data/Molmo-72B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Load conversations with error handling
try:
    with open("/localfolder/code/LLaMA-Factory/scripts/conversations.json", "r") as f:
        conversations = json.load(f)    
    logger.info(f"Loaded {len(conversations)} conversations")
except Exception as e:
    logger.error(f"Failed to load conversations: {e}")
    raise

batch_size = 64
failed_samples = []  # Keep track of failed samples
processed_count = 0
total_count = len(conversations)

# Process conversations in batches
for i in range(0, len(conversations), batch_size):
    batch_conversations = conversations[i:i+batch_size]
    all_processed_images = []
    
    logger.info(f"Processing batch {i//batch_size + 1}, samples {i+1}-{min(i+batch_size, total_count)}")
    
    # Prepare batch data by processing each sample individually then concatenating
    batch_inputs_list = []
    batch_image_paths = []
    batch_images = []
    valid_indices = []  # Track which samples in batch are valid
    
    # First round: Process each sample with error handling
    for j, conversation in enumerate(batch_conversations):
        try:
            image_path = conversation["images"][0].replace("/home/unitree/remote_jensen", "/localfolder/code/LLaMA-Factory/data/data")
            text = conversation["messages"][0]["content"]
            
            # Try to open and process the image
            try:
                image = Image.open(image_path)
            except Exception as e:
                logger.error(f"Failed to open image {image_path}: {e}")
                failed_samples.append({"batch": i, "index": j, "error": f"Image open error: {e}"})
                continue
            
            # Process each sample individually
            try:
                inputs = processor.process(
                    images=[image],
                    text=text
                )
                inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
                batch_inputs_list.append(inputs)
                batch_image_paths.append(image_path)
                batch_images.append(image)
                valid_indices.append(j)
            except Exception as e:
                logger.error(f"Failed to process sample {i+j+1}: {e}")
                failed_samples.append({"batch": i, "index": j, "error": f"Processing error: {e}"})
                continue
                
        except Exception as e:
            logger.error(f"Failed to prepare sample {i+j+1}: {e}")
            failed_samples.append({"batch": i, "index": j, "error": f"Preparation error: {e}"})
            continue
    
    # Skip this batch if no valid samples
    if not batch_inputs_list:
        logger.warning(f"No valid samples in batch {i//batch_size + 1}, skipping")
        continue
    
    try:
        # Concatenate individual inputs into a batch
        batch_inputs = {}
        for key in batch_inputs_list[0].keys():
            try:
                if key == 'input_ids':
                    # Concatenate input_ids
                    batch_inputs[key] = torch.cat([inp[key] for inp in batch_inputs_list], dim=0)
                elif key == 'images':
                    # Stack images
                    batch_inputs[key] = torch.cat([inp[key] for inp in batch_inputs_list], dim=0)
                elif key == 'image_input_idx':
                    # Concatenate image input indices
                    batch_inputs[key] = torch.cat([inp[key] for inp in batch_inputs_list], dim=0)
                else:
                    # For other keys, try to concatenate
                    batch_inputs[key] = torch.cat([inp[key] for inp in batch_inputs_list], dim=0)
            except Exception as e:
                logger.error(f"Failed to concatenate key {key}: {e}")
                # Try to continue with other keys or use individual processing
                batch_inputs[key] = batch_inputs_list[0][key]  # Use first sample as fallback
        
        # Move inputs to the correct device
        try:
            batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
        except Exception as e:
            logger.error(f"Failed to move inputs to device: {e}")
            continue
        
        # Generate output for the batch
        try:
            with torch.no_grad():
                output = model.generate_from_batch(
                    batch_inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>", temperature=0.5, do_sample=False),
                    tokenizer=processor.tokenizer,
                    use_cache=True
                )
        except Exception as e:
            logger.error(f"Failed to generate output for batch {i//batch_size + 1}: {e}")
            # Try individual processing for this batch
            logger.info("Attempting individual processing for failed batch")
            for k, (image_path, original_image, individual_inputs, valid_idx) in enumerate(zip(batch_image_paths, batch_images, batch_inputs_list, valid_indices)):
                try:
                    individual_inputs = {k: v.to(model.device) for k, v in individual_inputs.items()}
                    with torch.no_grad():
                        individual_output = model.generate_from_batch(
                            individual_inputs,
                            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>", temperature=0.5, do_sample=False),
                            tokenizer=processor.tokenizer,
                            use_cache=True
                        )
                    # Process individual result
                    input_length = individual_inputs['input_ids'].size(1)
                    generated_tokens = individual_output[0, input_length:]
                    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    conversation = batch_conversations[valid_idx]
                    conversation["messages"].append({
                        "role": "assistant",
                        "content": generated_text
                    })
                    
                    # Extract coordinates with error handling
                    try:
                        match = re.search(r'x1="([\d\.]+)" y1="([\d\.]+)" x2="([\d\.]+)" y2="([\d\.]+)"', generated_text)
                        
                        if match:
                            try:
                                x1, y1, x2, y2 = map(float, match.groups())
                                print(f"  Points: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                                
                                # Draw points on the image
                                image = original_image.copy()
                                width, height = image.size
                                x1_px, y1_px = int(x1 / 100 * width), int(y1 / 100 * height)
                                x2_px, y2_px = int(x2 / 100 * width), int(y2 / 100 * height)

                                conversation["points"] = [x1_px, y1_px, x2_px, y2_px]
                                
                                try:
                                    draw = ImageDraw.Draw(image)
                                    radius = 5
                                    color = (255, 0, 0)  # Red color
                                    
                                    # Draw circles at the two points
                                    draw.ellipse((x1_px - radius, y1_px - radius, x1_px + radius, y1_px + radius), 
                                                outline=color, width=radius)
                                    draw.ellipse((x2_px - radius, y2_px - radius, x2_px + radius, y2_px + radius), 
                                                outline=color, width=radius)
                                    
                                    all_processed_images.append(image)
                                except Exception as e:
                                    logger.error(f"Failed to draw points on image {i+valid_idx+1}: {e}")
                                    all_processed_images.append(original_image.copy())
                                    
                            except (ValueError, TypeError) as e:
                                logger.error(f"Failed to parse coordinates for image {i+valid_idx+1}: {e}")
                                print(f"  Failed to parse coordinates: {e}")
                                all_processed_images.append(original_image.copy())
                                conversation["points"] = []
                        else:
                            print(f"  No points found in generated text")
                            all_processed_images.append(original_image.copy())
                            conversation["points"] = []
                            
                    except Exception as e:
                        logger.error(f"Failed to process coordinates for image {i+valid_idx+1}: {e}")
                        all_processed_images.append(original_image.copy())
                        conversation["points"] = []
                        
                    processed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed individual processing for sample {i+valid_idx+1}: {e}")
                    failed_samples.append({"batch": i, "index": valid_idx, "error": f"Individual processing error: {e}"})
                    all_processed_images.append(original_image.copy())
            continue
        
        # Process each output in the batch
        for j, (image_path, original_image, individual_inputs, valid_idx) in enumerate(zip(batch_image_paths, batch_images, batch_inputs_list, valid_indices)):
            try:
                # Get generated tokens for this sample
                input_length = individual_inputs['input_ids'].size(1)
                generated_tokens = output[j, input_length:]
                generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                print(f"Image {i+valid_idx+1}: {generated_text}")
                
                conversation = batch_conversations[valid_idx]
                conversation["messages"].append({
                    "role": "assistant",
                    "content": generated_text
                })
                
                # Extract coordinates with error handling
                try:
                    match = re.search(r'x1="([\d\.]+)" y1="([\d\.]+)" x2="([\d\.]+)" y2="([\d\.]+)"', generated_text)
                    
                    if match:
                        try:
                            x1, y1, x2, y2 = map(float, match.groups())
                            print(f"  Points: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                            
                            # Draw points on the image
                            image = original_image.copy()
                            width, height = image.size
                            x1_px, y1_px = int(x1 / 100 * width), int(y1 / 100 * height)
                            x2_px, y2_px = int(x2 / 100 * width), int(y2 / 100 * height)

                            conversation["points"] = [x1_px, y1_px, x2_px, y2_px]
                            
                            try:
                                draw = ImageDraw.Draw(image)
                                radius = 5
                                color = (255, 0, 0)  # Red color
                                
                                # Draw circles at the two points
                                draw.ellipse((x1_px - radius, y1_px - radius, x1_px + radius, y1_px + radius), 
                                            outline=color, width=radius)
                                draw.ellipse((x2_px - radius, y2_px - radius, x2_px + radius, y2_px + radius), 
                                            outline=color, width=radius)
                                
                                all_processed_images.append(image)
                            except Exception as e:
                                logger.error(f"Failed to draw points on image {i+valid_idx+1}: {e}")
                                all_processed_images.append(original_image.copy())
                                
                        except (ValueError, TypeError) as e:
                            logger.error(f"Failed to parse coordinates for image {i+valid_idx+1}: {e}")
                            print(f"  Failed to parse coordinates: {e}")
                            all_processed_images.append(original_image.copy())
                            conversation["points"] = []
                    else:
                        print(f"  No points found in generated text")
                        all_processed_images.append(original_image.copy())
                        conversation["points"] = []
                        
                except Exception as e:
                    logger.error(f"Failed to process coordinates for image {i+valid_idx+1}: {e}")
                    all_processed_images.append(original_image.copy())
                    conversation["points"] = []
                    
                processed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process output for image {i+valid_idx+1}: {e}")
                failed_samples.append({"batch": i, "index": valid_idx, "error": f"Output processing error: {e}"})
                # Add original image as fallback
                try:
                    all_processed_images.append(original_image.copy())
                except:
                    # Create a blank image if even copying fails
                    all_processed_images.append(Image.new('RGB', (224, 224), (128, 128, 128)))

    except Exception as e:
        logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        continue

    # Second round: Ask model to judge if the marked points are on the end-effector
    try:
        print("\n=== Starting second round: End-effector validation ===")
        
        # Collect conversations that have valid points for second round
        valid_conversations = []
        valid_images_with_points = []
        valid_conversation_indices = []
        
        for j, (conversation, valid_idx) in enumerate(zip([batch_conversations[idx] for idx in valid_indices], valid_indices)):
            try:
                if "points" in conversation and len(conversation["points"]) == 4:
                    valid_conversations.append(conversation)
                    valid_images_with_points.append(all_processed_images[j])
                    valid_conversation_indices.append(valid_idx)
            except Exception as e:
                logger.error(f"Error checking points for conversation {i+valid_idx+1}: {e}")
                continue
        
        if valid_conversations:
            try:
                # Prepare second round batch
                second_round_inputs_list = []
                second_round_prompt = "Look at the red circles marked in this image. Are these two red circles both positioned on the robot's end-effector? Important: Both red dots must be on the end-effector. Please answer with 'yes' or 'no' only."
                
                for conversation, image_with_points in zip(valid_conversations, valid_images_with_points):
                    try:
                        # Process each image with the validation prompt
                        inputs = processor.process(
                            images=[image_with_points],
                            text=second_round_prompt
                        )
                        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
                        second_round_inputs_list.append(inputs)
                    except Exception as e:
                        logger.error(f"Failed to process second round input: {e}")
                        continue
                
                if second_round_inputs_list:
                    try:
                        # Concatenate second round inputs into a batch
                        second_batch_inputs = {}
                        for key in second_round_inputs_list[0].keys():
                            try:
                                if key == 'input_ids':
                                    second_batch_inputs[key] = torch.cat([inp[key] for inp in second_round_inputs_list], dim=0)
                                elif key == 'images':
                                    second_batch_inputs[key] = torch.cat([inp[key] for inp in second_round_inputs_list], dim=0)
                                elif key == 'image_input_idx':
                                    second_batch_inputs[key] = torch.cat([inp[key] for inp in second_round_inputs_list], dim=0)
                                else:
                                    second_batch_inputs[key] = torch.cat([inp[key] for inp in second_round_inputs_list], dim=0)
                            except Exception as e:
                                logger.error(f"Failed to concatenate second round key {key}: {e}")
                                second_batch_inputs[key] = second_round_inputs_list[0][key]
                        
                        # Move inputs to device
                        second_batch_inputs = {k: v.to(model.device) for k, v in second_batch_inputs.items()}
                        
                        # Generate second round output
                        try:
                            with torch.no_grad():
                                second_output = model.generate_from_batch(
                                    second_batch_inputs,
                                    GenerationConfig(max_new_tokens=50, stop_strings="<|endoftext|>", temperature=0.1, do_sample=False),
                                    tokenizer=processor.tokenizer,
                                    use_cache=True
                                )
                        except Exception as e:
                            logger.error(f"Failed second round generation: {e}")
                            # Continue without second round validation
                            second_output = None
                        
                        # Process second round outputs
                        if second_output is not None:
                            for j, (conversation, individual_inputs) in enumerate(zip(valid_conversations, second_round_inputs_list)):
                                try:
                                    input_length = individual_inputs['input_ids'].size(1)
                                    generated_tokens = second_output[j, input_length:]
                                    validation_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()
                                    
                                    # Extract yes/no answer
                                    is_end_effector = "yes" in validation_text and "no" not in validation_text
                                    
                                    # Add second round message to conversation
                                    conversation["messages"].append({
                                        "role": "user", 
                                        "content": second_round_prompt
                                    })
                                    conversation["messages"].append({
                                        "role": "assistant",
                                        "content": validation_text
                                    })
                                    conversation["is_end_effector"] = is_end_effector
                                    
                                    conv_idx = valid_conversation_indices[j]
                                    print(f"Validation for image {i+conv_idx+1}: {validation_text} -> {is_end_effector}")
                                except Exception as e:
                                    logger.error(f"Failed to process second round output {j}: {e}")
                                    conversation["is_end_effector"] = False
                                    continue
                                    
                    except Exception as e:
                        logger.error(f"Failed second round batch processing: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to prepare second round: {e}")
    except Exception as e:
        logger.error(f"Second round processing failed: {e}")
    
    # Create a grid of images for all processed images
    try:
        if all_processed_images:
            # Calculate grid dimensions
            num_images = len(all_processed_images)
            cols = 4  # 4 columns
            rows = math.ceil(num_images / cols)
            
            # Assuming all images have the same size (use first image dimensions)
            try:
                img_width, img_height = all_processed_images[0].size
                
                # Create the combined image
                combined_width = cols * img_width
                combined_height = rows * img_height
                combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
                
                # Paste each image into the grid
                for idx, img in enumerate(all_processed_images):
                    try:
                        row = idx // cols
                        col = idx % cols
                        x_offset = col * img_width
                        y_offset = row * img_height
                        combined_image.paste(img, (x_offset, y_offset))
                    except Exception as e:
                        logger.error(f"Failed to paste image {idx} into grid: {e}")
                        continue
                
                # Save the combined image
                try:
                    combined_image.save("batch_results_combined.png")
                    print(f"\nCombined image saved as 'batch_results_combined.png' with {num_images} images in a {rows}x{cols} grid")
                except Exception as e:
                    logger.error(f"Failed to save combined image: {e}")
                    
            except Exception as e:
                logger.error(f"Failed to create image grid: {e}")
                
    except Exception as e:
        logger.error(f"Failed to process image grid: {e}")
    
    # Save updated conversations periodically with error handling
    try:
        if i % 10 == 0:
            try:
                with open("/localfolder/code/LLaMA-Factory/scripts/conversations_with_validation.json", "w") as f:
                    json.dump(conversations, f, indent=2)
                print(f"\nUpdated conversations saved to 'conversations_with_validation.json'")
            except Exception as e:
                logger.error(f"Failed to save intermediate conversations: {e}")
    except Exception as e:
        logger.error(f"Error in periodic save: {e}")

# Final save with error handling
try:
    with open("/localfolder/code/LLaMA-Factory/scripts/conversations_with_validation.json", "w") as f:
        json.dump(conversations, f, indent=2)
    print(f"\nFinal conversations saved to 'conversations_with_validation.json'")
except Exception as e:
    logger.error(f"Failed to save final conversations: {e}")

# Print summary statistics
try:
    total_images = len(conversations)
    images_with_points = len([c for c in conversations if "points" in c and len(c.get("points", [])) == 4])
    images_with_end_effector = len([c for c in conversations if c.get("is_end_effector", False)])

    print(f"\nSummary:")
    print(f"Total images processed: {processed_count}/{total_images}")
    print(f"Images with valid points detected: {images_with_points}")
    print(f"Images where points are on end-effector: {images_with_end_effector}")
    print(f"Failed samples: {len(failed_samples)}")
    
    if images_with_points > 0:
        accuracy_rate = images_with_end_effector / images_with_points * 100
        print(f"End-effector detection rate: {accuracy_rate:.1f}%")
        
    if failed_samples:
        print(f"\nFailed samples details:")
        for failure in failed_samples[:10]:  # Show first 10 failures
            print(f"  Batch {failure['batch']}, Index {failure['index']}: {failure['error']}")
        if len(failed_samples) > 10:
            print(f"  ... and {len(failed_samples) - 10} more failures")
            
except Exception as e:
    logger.error(f"Failed to generate summary: {e}")

logger.info("Script completed")
                