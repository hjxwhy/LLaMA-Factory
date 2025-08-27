import sys
import os
import re
import torch

sys.path.append("/localfolder/code/LLaMA-Factory")

from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2ForCausalLM

from src.llamafactory.model.configuration_dinotxt_qwen2_5_vl import DinotxtQwen2_5_VLConfig
from src.llamafactory.model.modeling_dinotxt_qwen2_5_vl import DinotxtQwen2_5_VLForConditionalGeneration


def split_qkv(state_dict: dict):
    keys = [x for x in state_dict.keys() if "qkv" in x]
    for key in keys:
        qkv = state_dict.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        state_dict[key.replace("qkv", "q_proj")] = q
        state_dict[key.replace("qkv", "k_proj")] = k
        state_dict[key.replace("qkv", "v_proj")] = v
    return state_dict

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Vision head mappings for visual_state_dict -> vision_head conversion
    r"blocks.(\d+).attn.proj":      r"blocks.\1.attention.o_proj",
    r"blocks.(\d+).attn.q_proj":    r"blocks.\1.attention.q_proj",
    r"blocks.(\d+).attn.k_proj":    r"blocks.\1.attention.k_proj",
    r"blocks.(\d+).attn.v_proj":    r"blocks.\1.attention.v_proj",
    r"blocks.(\d+).ls1.gamma":      r"blocks.\1.layer_scale1.lambda1",
    r"blocks.(\d+).ls2.gamma":      r"blocks.\1.layer_scale2.lambda1",
    r"blocks.(\d+).mlp":            r"blocks.\1.ffn",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict

if __name__ == "__main__":
    # dinov3_vitl16_pretrain_lvd1689m = "/localfolder/data/dinov3/hf_models/vitl16_lvd1689m"
    # model = DINOv3ViTModel.from_pretrained(dinov3_vitl16_pretrain_lvd1689m)
    # dinov3_config = model.config
    # # print(dinov3_config)

    qwen2_5_vl_path = "/DATA/disk0/Qwen2.5-VL-7B-Instruct"
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen2_5_vl_path, device_map="auto", torch_dtype="bfloat16", attn_implementation="flash_attention_2",)
    
    # language_model = qwen.model.language_model
    # language_model_config = language_model.config
    # print(language_model_config.to_dict())
    
    # from src.llamafactory.model.dinotxt_head import VisionHeadConfig, VisionHead
    # head_config = {
    #     "input_dim": dinov3_config.hidden_size,
    #     "embed_dim": 2048,
    #     "num_heads": dinov3_config.num_attention_heads,
    #     "num_blocks": 2,
    #     "blocks_drop_path": 0.0,
    # }
    # head_config = VisionHeadConfig(**head_config)
    # vision_head = VisionHead(head_config)

    # dinotxt = "/localfolder/data/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
    # state_dict = torch.load(dinotxt)
    # visual_state_dict = {}
    # for k, v in state_dict.items():
    #     if "visual_model" in k:
    #         visual_state_dict[k.replace("visual_model.head.", "")] = v.clone()
    # del state_dict

    # visual_state_dict = split_qkv(visual_state_dict)
    
    # # Test the key conversion
    # visual_keys = list(visual_state_dict.keys())
    
    # converted_keys = convert_old_keys_to_new_keys(visual_keys)
    # print("Key conversion test:")
    # for old_key, new_key in converted_keys.items():
    #     if old_key != new_key:
    #         print(f"  {old_key} -> {new_key}")
    
    # converted_state_dict = {}
    # for key in visual_keys:
    #     new_key = converted_keys[key]
    #     weight_tensor = visual_state_dict[key]
    #     converted_state_dict[new_key] = weight_tensor
    # vision_head.load_state_dict(converted_state_dict, strict=True)

    # dinotxt_qwen2_5_vl_config = DinotxtQwen2_5_VLConfig(vision_config=dinov3_config.to_dict(), text_config=language_model_config.to_dict(), head_config=head_config.to_dict())
    # print(dinotxt_qwen2_5_vl_config.to_dict())

    # dinotxt_qwen2_5_vl = DinotxtQwen2_5_VLForConditionalGeneration(dinotxt_qwen2_5_vl_config, vision_model=model, language_model=language_model, vision_head=vision_head)
    # print(dinotxt_qwen2_5_vl)

    # # breakpoint()
    # dinotxt_qwen2_5_vl.save_pretrained("/localfolder/data/dinotxt_qwen2_5_vl")
    dinotxt_qwen2_5_vl_path = "/localfolder/data/dinotxt_qwen2_5_vl"
    dinotxt_qwen2_5_vl = DinotxtQwen2_5_VLForConditionalGeneration.from_pretrained(dinotxt_qwen2_5_vl_path, device_map="auto", torch_dtype="bfloat16", attn_implementation="flash_attention_2",)
    # dinotxt_qwen2_5_vl.to("cuda")
    # # breakpoint()

    qwen2_5_vl_path = "/DATA/disk0/Qwen2.5-VL-7B-Instruct"
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(qwen2_5_vl_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello who are you?"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = None, None
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    # generated_ids = dinotxt_qwen2_5_vl.generate(**inputs, max_new_tokens=128)
    breakpoint()

