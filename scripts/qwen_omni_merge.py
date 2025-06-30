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

import os
import shutil

import fire
import torch
from peft import PeftModel
from transformers import AutoModel, AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniForConditionalGeneration  # type: ignore


def merge_lora(
    base_model_path: str,
    lora_checkpoint_path: str,
    extra_file: str = "spk_dict.pt",
    submodule_name: str = "thinker",
    save_path: str = "./merged_model_checkpoint",
    auto_verify: bool = True,
    tolerance: float = 1e-6,
):
    """Load the original model, tokenizer, and processor configuration, merge the LoRA weights.

    For a specified submodule, and save the final merged model along with its configurations.

    Args:
        base_model_path (str): Path to the original model directory.
        lora_checkpoint_path (str): Path to the directory containing LoRA weights.
        extra_file (str): Name of the extra file to be copied (default: "spk_dict.pt").
        submodule_name (str): Name of the submodule to merge (default: "thinker").
        save_path (str): Directory where the merged model and configurations will be saved.
        auto_verify (bool): Whether to automatically verify the merged model (default: True).
        tolerance (float): Tolerance for weight comparison during verification (default: 1e-6).
    """
    # 1. Load the original model, tokenizer, and processor
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")
    processor = AutoProcessor.from_pretrained(base_model_path)
    print("Successfully loaded the original model and tokenizer.")

    # 2. Extract the submodule to be merged (e.g., model.thinker)
    if not hasattr(model, submodule_name):
        raise AttributeError(f"The model does not have a submodule named '{submodule_name}'.")

    base_submodule = getattr(model, submodule_name)
    print(f"Successfully extracted submodule: {submodule_name}.")

    # Store original submodule state for verification
    if auto_verify:
        original_submodule_state = base_submodule.state_dict()

    # 3. Load the LoRA weights onto the extracted submodule
    lora_model = PeftModel.from_pretrained(base_submodule, lora_checkpoint_path)
    print("LoRA weights loaded successfully.")

    # 4. Merge the LoRA weights into the submodule and unload the LoRA modules
    merged_submodule = lora_model.merge_and_unload()
    print("LoRA weights merged successfully.")

    # 5. Replace the original submodule with the merged submodule in the model
    setattr(model, submodule_name, merged_submodule)

    # 6. Save the final merged model along with the tokenizer and processor configuration
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Merged model and tokenizer saved to {save_path}.")

    source_file = os.path.join(base_model_path, extra_file)
    target_file = os.path.join(save_path, extra_file)
    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
        print(f"File '{extra_file}' copied from {base_model_path} to {save_path}.")
    else:
        print(f"File '{extra_file}' not found in {base_model_path}, skipping copy.")

    # 7. Auto-verify the merged model if requested
    if auto_verify:
        print("\n" + "="*50)
        print("🔍 AUTO-VERIFICATION: Checking LoRA merge...")
        print("="*50)
        
        # Load the saved model and compare its thinker with the merged submodule
        try:
            saved_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                save_path, torch_dtype="auto", device_map="cpu"
            )
            saved_submodule = getattr(saved_model, submodule_name)
            
            # Compare the merged submodule with the saved submodule
            merged_state = merged_submodule.state_dict()
            saved_state = saved_submodule.state_dict()
            
            # Check if all keys match
            if set(merged_state.keys()) != set(saved_state.keys()):
                print("ERROR: State dict keys don't match between merged and saved model!")
                return False
            
            # Compare weights
            all_weights_match = True
            mismatched_layers = []
            
            for key in merged_state.keys():
                merged_weight = merged_state[key]
                saved_weight = saved_state[key]
                
                if merged_weight.shape != saved_weight.shape:
                    print(f"ERROR: Shape mismatch for {key}")
                    all_weights_match = False
                    mismatched_layers.append(key)
                    continue
                
                if not torch.allclose(merged_weight, saved_weight, atol=tolerance, rtol=tolerance):
                    max_diff = torch.max(torch.abs(merged_weight - saved_weight)).item()
                    print(f"ERROR: Weight mismatch for {key}: max_diff={max_diff}")
                    all_weights_match = False
                    mismatched_layers.append(key)
                else:
                    print(f"✓ {key}: weights match between merged and saved model")
            
            print("="*50)
            if all_weights_match:
                print("🎉 LORA MERGE VERIFICATION: PASSED")
                print("   All weights match between merged model and saved model!")
            else:
                print("⚠️  LORA MERGE VERIFICATION: FAILED")
                print(f"   {len(mismatched_layers)} layers have mismatched weights")
            print("="*50)
            
            return all_weights_match
            
        except Exception as e:
            print(f"ERROR during verification: {e}")
            print("="*50)
            print("⚠️  LORA MERGE VERIFICATION: FAILED")
            print("="*50)
            return False
    
    return True


def save_full_model(
    saved_thinker_path: str,
    base_model_path: str,
    save_path: str = "./merged_model_checkpoint",
    extra_file: str = "spk_dict.pt",
    auto_verify: bool = True,
    tolerance: float = 1e-6,
):
    """Load the saved thinker module and the original model, replace the thinker in the original model.

    Then save the complete model along with its tokenizer and processor configuration.

    Args:
        saved_thinker_path (str): Path to the saved thinker weights.
        base_model_path (str): Directory path of the original model.
        save_path (str): Directory where the merged model and configurations will be saved.
        extra_file (str): Name of the extra file to be copied (default: "spk_dict.pt").
        auto_verify (bool): Whether to automatically verify the saved model (default: True).
        tolerance (float): Tolerance for weight comparison during verification (default: 1e-6).
    """
    # 1. Load the saved thinker module and the original model
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        saved_thinker_path, torch_dtype="auto", device_map="cpu"
    )
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")
    base_model.thinker = thinker

    # 2. Save the complete model along with its tokenizer and processor configuration
    processor = AutoProcessor.from_pretrained(base_model_path)
    base_model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Merged model and tokenizer saved to {save_path}.")

    # 3. Copy the extra file from the base model directory to the save_path
    source_file = os.path.join(base_model_path, extra_file)
    target_file = os.path.join(save_path, extra_file)
    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
        print(f"File '{extra_file}' copied from {base_model_path} to {save_path}.")
    else:
        print(f"File '{extra_file}' not found in {base_model_path}, skipping copy.")

    # 4. Auto-verify the saved model if requested
    if auto_verify:
        print("\n" + "="*50)
        print("🔍 AUTO-VERIFICATION: Checking saved model...")
        print("="*50)
        
        verification_result = verify_model_weights(
            saved_full_model_path=save_path,
            original_thinker_path=saved_thinker_path,
            tolerance=tolerance
        )
        
        print("="*50)
        if verification_result:
            print("🎉 MODEL SAVE VERIFICATION: PASSED")
        else:
            print("⚠️  MODEL SAVE VERIFICATION: FAILED")
            print("   Please check the verification details above.")
        print("="*50)
        
        return verification_result
    
    return True


def verify_model_weights(
    saved_full_model_path: str,
    original_thinker_path: str,
    tolerance: float = 1e-6,
):
    """Verify that the saved full model's thinker weights match the original thinker weights.
    
    Args:
        saved_full_model_path (str): Path to the saved full model directory.
        original_thinker_path (str): Path to the original thinker model directory.
        tolerance (float): Tolerance for weight comparison (default: 1e-6).
    
    Returns:
        bool: True if weights match within tolerance, False otherwise.
    """
    print("Starting model weight verification...")
    
    # Load the saved full model
    try:
        saved_full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            saved_full_model_path, torch_dtype="auto", device_map="cpu"
        )
        print("Successfully loaded saved full model.")
    except Exception as e:
        print(f"Error loading saved full model: {e}")
        return False
    
    # Load the original thinker model
    try:
        original_thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            original_thinker_path, torch_dtype="auto", device_map="cpu"
        )
        print("Successfully loaded original thinker model.")
    except Exception as e:
        print(f"Error loading original thinker model: {e}")
        return False
    
    # Get the thinker from the saved full model
    saved_thinker = saved_full_model.thinker
    
    # Compare state dictionaries
    saved_thinker_state = saved_thinker.state_dict()
    original_thinker_state = original_thinker.state_dict()
    
    # Check if all keys match
    if set(saved_thinker_state.keys()) != set(original_thinker_state.keys()):
        print("ERROR: State dict keys don't match!")
        missing_in_saved = set(original_thinker_state.keys()) - set(saved_thinker_state.keys())
        missing_in_original = set(saved_thinker_state.keys()) - set(original_thinker_state.keys())
        
        if missing_in_saved:
            print(f"Missing in saved model: {missing_in_saved}")
        if missing_in_original:
            print(f"Extra in saved model: {missing_in_original}")
        return False
    
    # Compare weights
    all_weights_match = True
    mismatched_layers = []
    
    for key in saved_thinker_state.keys():
        saved_weight = saved_thinker_state[key]
        original_weight = original_thinker_state[key]
        
        # Check if shapes match
        if saved_weight.shape != original_weight.shape:
            print(f"ERROR: Shape mismatch for {key}: saved={saved_weight.shape}, original={original_weight.shape}")
            all_weights_match = False
            mismatched_layers.append(key)
            continue
        
        # Check if values match within tolerance
        if not torch.allclose(saved_weight, original_weight, atol=tolerance, rtol=tolerance):
            max_diff = torch.max(torch.abs(saved_weight - original_weight)).item()
            print(f"ERROR: Weight mismatch for {key}: max_diff={max_diff}")
            all_weights_match = False
            mismatched_layers.append(key)
        else:
            print(f"✓ {key}: weights match within tolerance")
    
    if all_weights_match:
        print("✅ SUCCESS: All thinker weights match between saved model and original thinker!")
        return True
    else:
        print(f"❌ FAILED: {len(mismatched_layers)} layers have mismatched weights:")
        for layer in mismatched_layers[:10]:  # Show first 10 mismatched layers
            print(f"  - {layer}")
        if len(mismatched_layers) > 10:
            print(f"  ... and {len(mismatched_layers) - 10} more")
        return False


if __name__ == "__main__":
    fire.Fire({"save_full": save_full_model, "merge_lora": merge_lora, "verify": verify_model_weights})
