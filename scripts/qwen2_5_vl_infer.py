import os, re, json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
from datetime import datetime
from typing import List, Dict, Any, Union


class UnifiedInference:
    def __init__(self, model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, device_map=device_map, attn_implementation=attn_implementation)
        self.model.eval()
        # Try to load a default font, fallback to PIL default if not available
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
        except:
            try:
                self.font = ImageFont.truetype("arial.ttf", 20)
            except:
                self.font = ImageFont.load_default()
    
    def _preprocess_image(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image
    
    def _preprocess_images(self, images):
        """Process multiple images"""
        processed_images = []
        for img in images:
            processed_images.append(self._preprocess_image(img))
        return processed_images
    
    def draw_bbox_on_image(self, image, coord, output_path):
        """Draw bounding box on image"""
        img_with_bbox = image.copy()
        draw = ImageDraw.Draw(img_with_bbox)
        draw.rectangle([(coord[0], coord[1]), (coord[2], coord[3])], outline="red", width=3)
        img_with_bbox.save(output_path)
        print(f"Saved image with bounding box to {output_path}")

    def draw_trajectory_on_image(self, image, coord, output_path):
        """Draw trajectory points on image"""
        img_with_points = image.copy()
        draw = ImageDraw.Draw(img_with_points)
        
        # Draw each trajectory point
        for point in coord:
            x, y = point
            draw.ellipse([(x-3, y-3), (x+3, y+3)], fill="red", outline="red")
        
        # Connect points with lines to show trajectory
        if len(coord) > 1:
            draw.line([tuple(point) for point in coord], fill="yellow", width=2)
        
        img_with_points.save(output_path)
        print(f"Saved image with trajectory points to {output_path}")

    def draw_points_on_image(self, image, coord, output_path):
        """Draw points on image"""
        img_with_points = image.copy()
        draw = ImageDraw.Draw(img_with_points)
        
        # Handle different coordinate formats
        if isinstance(coord, list) and len(coord) > 0:
            if isinstance(coord[0], dict):
                points = [list(v.values()) for v in coord]
            elif isinstance(coord[0], list):
                points = coord
            else:
                points = [coord]
        elif isinstance(coord, dict):
            points = [list(coord.values())]
        else:
            points = [coord]
        
        # Draw each point
        for i, point in enumerate(points):
            x, y = point
            draw.ellipse([(x-5, y-5), (x+5, y+5)], fill="red", outline="red")
            # Add number labels
            draw.text((x+8, y-8), str(i+1), fill="blue")
        
        img_with_points.save(output_path)
        print(f"Saved image with points to {output_path}")

    def draw_task_infer_on_image(self, image, coord, output_path):
        """Draw task_infer points with labels on image"""
        img_with_points = image.copy()
        draw = ImageDraw.Draw(img_with_points)
        
        # Handle task_infer format: [{"point": [x, y], "label": "label"}, ...]
        if not isinstance(coord, list):
            coord = [coord]

        for i, item in enumerate(coord):
            point = item["point"]
            label = item["label"]
            x, y = point
            
            # Draw point
            draw.ellipse([(x-5, y-5), (x+5, y+5)], fill="green", outline="green")
            
            # Draw label
            draw.text((x+10, y-5), f"{i+1}: {label}", fill="blue", font=self.font)
        
        img_with_points.save(output_path)
        print(f"Saved image with task_infer points to {output_path}")

    def draw_text_on_image(self, image, text, output_path, max_width=None):
        """Draw text on image with proper line wrapping"""
        img_with_text = image.copy()
        width, height = img_with_text.size
        
        # Calculate max width for text if not provided
        if max_width is None:
            max_width = width - 40  # Leave 20px margin on each side
        
        # Split text into lines that fit within max_width
        lines = []
        words = text.split(' ')
        current_line = ""
        
        draw = ImageDraw.Draw(img_with_text)
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=self.font)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Calculate text area height
        line_height = 25
        text_height = len(lines) * line_height + 20  # 20px padding
        
        # Extend image if necessary to fit text
        if text_height > height * 0.3:  # If text takes more than 30% of image height
            new_height = height + text_height
            new_img = Image.new('RGB', (width, new_height), color='white')
            new_img.paste(img_with_text, (0, 0))
            img_with_text = new_img
            draw = ImageDraw.Draw(img_with_text)
        
        # Draw text lines
        y_offset = height + 10 if img_with_text.size[1] > height else height - text_height + 10
        for i, line in enumerate(lines):
            draw.text((20, y_offset + i * line_height), line, fill="black", font=self.font)
        
        img_with_text.save(output_path)
        print(f"Saved image with text overlay to {output_path}")

    def _get_output_type(self, task):
        """Determine output type based on task"""
        if task in ["affordance", "grounding"]:
            return "bbox"
        elif task == "trajectory":
            return "trajectory"
        elif task in ["pointing", "spatial_reasoning"]:
            return "point"
        elif task == "task_infer":
            return "task_infer"
        elif task in ["spatial_question", "task_planning", "choices"]:
            return "text"
        else:
            return "text"

    def _generate_unique_filename(self, output_dir, task_id, output_type, extension="jpg"):
        """Generate unique filename to avoid conflicts"""
        timestamp = datetime.now().strftime("%H%M%S")
        base_filename = f"{task_id}_{output_type}_{timestamp}.{extension}"
        return os.path.join(output_dir, base_filename)

    def parse_and_visualize_output(self, output_text, task, images, output_dir=None, task_id="result"):
        """Parse output text and create visualizations based on task type"""
        output_type = self._get_output_type(task)
        results = []
        
        # Save text output
        if output_dir:
            text_output_path = self._generate_unique_filename(output_dir, task_id, "output", "txt")
            with open(text_output_path, "w", encoding="utf-8") as f:
                f.write(f"Task: {task}\n")
                f.write(f"Answer: {output_text}\n")
        
        # Process visual outputs based on type
        if output_type in ["bbox", "trajectory", "point", "task_infer"] and output_dir and images:
            try:
                # Try to parse the coordinates
                coord = json.loads(output_text)
                
                # Process each image (in case of multiple images)
                for idx, image in enumerate(images):
                    suffix = f"_img{idx}" if len(images) > 1 else ""
                    
                    if output_type == "bbox" and len(coord) == 4:
                        # Bounding box format: [x1, y1, x2, y2]
                        image_output_path = self._generate_unique_filename(output_dir, f"{task_id}{suffix}", "bbox")
                        self.draw_bbox_on_image(image, coord, image_output_path)
                        results.append(image_output_path)
                        
                    elif output_type == "trajectory" and isinstance(coord, list):
                        # Trajectory format: [[x1, y1], [x2, y2], ...]
                        if isinstance(coord[0], dict):
                            coord = [list(v.values()) for v in coord]
                        image_output_path = self._generate_unique_filename(output_dir, f"{task_id}{suffix}", "trajectory")
                        self.draw_trajectory_on_image(image, coord, image_output_path)
                        results.append(image_output_path)
                        
                    elif output_type == "point":
                        # Point(s) format: {"x": x, "y": y} or [{"x": x, "y": y}, ...]
                        image_output_path = self._generate_unique_filename(output_dir, f"{task_id}{suffix}", "points")
                        self.draw_points_on_image(image, coord, image_output_path)
                        results.append(image_output_path)
                        
                    elif output_type == "task_infer":
                        # Task infer format: [{"point": [x, y], "label": "label"}, ...]
                        image_output_path = self._generate_unique_filename(output_dir, f"{task_id}{suffix}", "task_infer")
                        self.draw_task_infer_on_image(image, coord, image_output_path)
                        results.append(image_output_path)
                    
            except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
                print(f"Could not parse coordinates for visualization: {e}")
                print(f"Raw output: {output_text}")
        
        # Handle pure text outputs
        elif output_type == "text" and output_dir and images:
            for idx, image in enumerate(images):
                suffix = f"_img{idx}" if len(images) > 1 else ""
                image_output_path = self._generate_unique_filename(output_dir, f"{task_id}{suffix}", "text")
                self.draw_text_on_image(image, output_text, image_output_path)
                results.append(image_output_path)
        
        return output_text, output_type, results

    @torch.inference_mode()
    def inference(self, images, text, max_new_tokens=100, temperature=0.5, do_sample=True, top_k=None, task=None, 
                  output_dir=None, task_id="result", num_beams=1, min_new_tokens=1, top_p=0.9):
        
        # Support both single image and multiple images
        if isinstance(images, (str, np.ndarray)) or hasattr(images, 'size'):
            images = [images]
        
        processed_images = self._preprocess_images(images)

        if task=="pointing":
            text = f"Point to any {text} in the image and format output as JSON with x, y coordinates"
        elif task=="affordance":
            text = f"You are a robot using the joint control. The task is <{text}>. Please predict a possible affordance area of the end effector, delivering coordinates in plain text format \"[x1, y1, x2, y2]\"."
        elif task=="trajectory":
            text = f"You are a robot using the joint control. The task is <{text}>. Please predict 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point."
        elif task=="spatial_reasoning":
            text = f"Point to the {text} in the image and format output as JSON with x, y coordinates"
        elif task=="spatial_question":
            text = text
        elif task=="task_planning":
            text = f"To work toward <{text}>, what are the next four steps?, Answer in the format 1-<>, 2-<>, ..."
        elif task=="task_infer":
            text = (
                "You need to help me point out the starting point and end point of the task on the image.\n"
                "For example, for this task: \"Put the spoon to the right of the cloth.\",\n"
                "First you need to determine that the starting location of this task is label: spoon, and the end location is label: the right of the cloth.\n"
                "Then point to the starting location and the end location on the image.\n"
                "The answer should follow the json format: [{\"point\": [x, y], \"label\": <label1>}, ...].\n"
                "The points are in [x, y] format.\n"
                f"Now your task is\n\"{text}.\"\n"
            )
        elif task=="grounding":
            text = f"Detect {text} in the image, delivering coordinates in plain text format \"[x1, y1, x2, y2]\"."
        elif task=="choices":
            text = "Given these two images, find the corresponding points between them.\nWhich labeled point in second image is the same as the circled point in first image?\nAnswer the question in single letter."
        else:
            raise ValueError(f"Invalid task: {task}")
        
        # For multiple images, use the first image for the main processing
        # You can modify this logic based on your specific multi-image requirements
        primary_image = processed_images[0]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": primary_image
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
        
        # Add additional images if present
        if len(processed_images) > 1:
            for img in processed_images[1:]:
                messages[0]["content"].insert(-1, {
                    "type": "image",
                    "image": img
                })
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, 
                                        min_new_tokens=min_new_tokens, top_p=top_p, top_k=top_k, 
                                        temperature=temperature, do_sample=do_sample)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse and visualize output if requested
        if output_dir or task:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            parsed_output, output_type, result_files = self.parse_and_visualize_output(
                output_text, task, processed_images, output_dir, task_id
            )
            outputs = {
                "text": output_text,
                "task": task,
                "task_id": task_id,
                "output_type": output_type,
                "parsed": parsed_output,
                "result_files": result_files
            }
            print(f"Output: {outputs}")
            return outputs
        
        return output_text

    def batch_inference(self, config_file_or_dict):
        """Run batch inference from configuration file or dictionary"""
        if isinstance(config_file_or_dict, str):
            with open(config_file_or_dict, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = config_file_or_dict
        
        output_dir = config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for task_config in config["tasks"]:
            task_id = task_config["id"]
            task = task_config["task"]
            images = task_config["images"]
            text = task_config["text"]
            
            print(f"\n{'='*50}")
            print(f"Processing task: {task_id} ({task})")
            print(f"Images: {images}")
            print(f"Text: {text}")
            print(f"{'='*50}")
            
            try:
                result = self.inference(
                    images=images,
                    text=text,
                    task=task,
                    output_dir=output_dir,
                    task_id=task_id,
                    max_new_tokens=task_config.get("max_new_tokens", 1000),
                    temperature=task_config.get("temperature", 0.5),
                    do_sample=task_config.get("do_sample", True),
                    top_k=task_config.get("top_k", None),
                    num_beams=task_config.get("num_beams", 1)
                )
                results.append(result)
                print(f"✅ Task {task_id} completed successfully")
                
            except Exception as e:
                print(f"❌ Task {task_id} failed: {str(e)}")
                results.append({
                    "task_id": task_id,
                    "task": task,
                    "error": str(e)
                })
        
        # Save summary results
        summary_path = os.path.join(output_dir, "batch_results_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 Batch processing completed! Summary saved to {summary_path}")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file for batch processing")
    parser.add_argument("--image", type=str, help="Single image path (for single inference)")
    parser.add_argument("--images", type=str, nargs='+', help="Multiple image paths (for single inference)")
    parser.add_argument("--text", type=str, help="Text input (for single inference)")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--task", type=str, help="Task type (for single inference)")
    parser.add_argument("--task_id", type=str, default="single_task", help="Task ID (for single inference)")
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    # Initialize the model
    model = UnifiedInference(args.model_name_or_path)

    # Batch processing mode
    if args.config:
        model.batch_inference(args.config)
    
    # Single inference mode
    elif args.task and (args.image or args.images) and args.text:
        images = args.images if args.images else [args.image]
        result = model.inference(
            images=images,
            text=args.text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_k=args.top_k,
            task=args.task,
            output_dir=args.output_dir,
            task_id=args.task_id,
            num_beams=args.num_beams
        )
    else:
        print("❌ Error: Either provide --config for batch processing, or provide --task, --text, and --image/--images for single inference")