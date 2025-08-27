import logging
from functools import partial
from typing import Optional, Tuple, Callable

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, auto_docstring, logging
from transformers.configuration_utils import PretrainedConfig

from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTAttention, DINOv3ViTLayerScale, DINOv3ViTDropPath, DINOv3ViTModel
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel, Qwen2_5_VLTextModel, Qwen2_5_VLForConditionalGeneration

from .configuration_dinotxt_qwen2_5_vl import DINOv3ViTQwen2_5_VLConfig, VisionTowerConfig, VisionHeadConfig

logger = logging.get_logger(__name__)



class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network implementation compatible with the original dinov3 implementation.
    
    Args:
        in_features (`int`): Input feature dimension
        hidden_features (`int`, *optional*): Hidden layer dimension. Defaults to in_features if None.
        out_features (`int`, *optional*): Output feature dimension. Defaults to in_features if None.
        act_layer (`Callable`, *optional*): Activation layer. Not used in SwiGLU as SiLU is hardcoded.
        drop (`float`, *optional*, defaults to 0.0): Dropout rate. Not used in this implementation.
        bias (`bool`, *optional*, defaults to True): Whether to use bias in linear layers.
        align_to (`int`, *optional*, defaults to 8): Alignment factor for hidden features.
        device: Device to place the module on.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block that mimics the original dinov3 SelfAttentionBlock using transformers components.
    
    Args:
        input_dim (`int`): Input dimension
        num_heads (`int`): Number of attention heads
        ffn_layer (`Callable`): FFN layer constructor (e.g., partial(SwiGLUFFN, align_to=64))
        init_values (`float`, *optional*, defaults to 1e-5): Layer scale initialization value
        drop_path (`float`, *optional*, defaults to 0.0): Drop path rate
        attn_implementation (`str`, *optional*, defaults to "sdpa"): Attention implementation type
    """
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_layer: Callable = None,
        ffn_ratio: float = 4.0,
        init_values: float = 1e-5,
        drop_path: float = 0.0,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        
        # Create a minimal DINOv3ViTConfig for attention
        attention_config = DINOv3ViTConfig(
            hidden_size=input_dim,
            num_attention_heads=num_heads,
            attention_dropout=0.0,
            layer_norm_eps=1e-6,
            query_bias=False,
            key_bias=False,
            value_bias=False,
            proj_bias=True,
            _attn_implementation=attn_implementation,
        )
        
        self.norm1 = nn.LayerNorm(input_dim, eps=1e-6)
        self.attention = DINOv3ViTAttention(attention_config)
        self.layer_scale1 = DINOv3ViTLayerScale(
            type('Config', (), {'layerscale_value': init_values, 'hidden_size': input_dim})()
        )
        
        self.drop_path = DINOv3ViTDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(input_dim, eps=1e-6)
        
        # Create FFN layer
        mlp_hidden_dim = int(input_dim * ffn_ratio)
        if ffn_layer is not None:
            self.ffn = ffn_layer(input_dim, mlp_hidden_dim, align_to=64)
        else:
            self.ffn = SwiGLUFFN(input_dim, mlp_hidden_dim, align_to=64)
        
        self.layer_scale2 = DINOv3ViTLayerScale(
            type('Config', (), {'layerscale_value': init_values, 'hidden_size': input_dim})()
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states,
            attention_mask=None,
            position_embeddings=(None, None),  # No RoPE embeddings
        )
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        # FFN with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states


VISION_HEAD_START_DOCSTRING = r"""
    Vision head module that processes visual tokens through attention blocks and applies linear projection.
    
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PretrainedConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VISION_TOWER_START_DOCSTRING = r"""
    Vision tower module that combines a backbone with a vision head for processing images.
    
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PretrainedConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""




@add_start_docstrings(VISION_HEAD_START_DOCSTRING)
class VisionHead(PreTrainedModel):
    """
    Vision head module that processes visual tokens through attention blocks and applies linear projection.
    """
    
    config_class = VisionHeadConfig
    config: VisionHeadConfig
    _supports_flash_attn = True
    _no_split_modules = ["SelfAttentionBlock"]
    
    def __init__(self, config: VisionHeadConfig):
        super().__init__(config)
        
        self.config = config
        
        # Initialize blocks
        block_list = [nn.Identity()]
        self.ln_final = nn.Identity()
        
        if config.num_blocks > 0:
            # Get attention implementation from config
            attn_impl = getattr(config, '_attn_implementation', 'sdpa')
            block_list = [
                SelfAttentionBlock(
                    config.input_dim,
                    config.num_heads,
                    ffn_layer=partial(SwiGLUFFN, align_to=64),
                    init_values=1e-5,
                    drop_path=config.blocks_drop_path,
                    attn_implementation=attn_impl,
                )
                for _ in range(config.num_blocks)
            ]
            self.ln_final = nn.LayerNorm(config.input_dim)
            
        self.blocks = nn.ModuleList(block_list)
        
        # Calculate multiplier based on token usage
        multiplier = 2 if config.use_class_token and config.use_patch_tokens else 1
        self.linear_projection = nn.Identity()
        
        if multiplier * config.input_dim != config.embed_dim or config.use_linear_projection:
            logger.info(
                f"Vision Head: Using a linear projection from {config.input_dim} to {config.embed_dim}"
            )
            assert config.embed_dim % multiplier == 0, (
                f"Expects {config.embed_dim} to be divisible by {multiplier}"
            )
            self.linear_projection = nn.Linear(
                config.input_dim, config.embed_dim // multiplier, bias=False
            )
            
        # Initialize weights
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=module.in_features**-0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def init_weights(self):
        """Initialize weights following the vision transformer initialization scheme."""
        super().init_weights()
        
        if self.config.num_blocks > 0:
            # Apply DINOv3-style initialization to each block
            for block in self.blocks:
                if isinstance(block, SelfAttentionBlock):
                    self._init_self_attention_block(block)
            if hasattr(self.ln_final, 'reset_parameters'):
                self.ln_final.reset_parameters()
                
        if isinstance(self.linear_projection, nn.Linear):
            nn.init.normal_(
                self.linear_projection.weight,
                std=self.linear_projection.in_features**-0.5,
            )
    
    def _init_self_attention_block(self, block: SelfAttentionBlock):
        """Initialize a SelfAttentionBlock using DINOv3-style initialization."""
        initializer_range = 0.02  # Standard initializer range
        
        # Initialize attention layers
        for module in block.attention.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight.data.to(torch.float32),
                    mean=0.0,
                    std=initializer_range,
                )
                if module.bias is not None:
                    module.bias.data.zero_()
        
        # Initialize FFN layers (SwiGLUFFN)
        for module in block.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight.data.to(torch.float32),
                    mean=0.0,
                    std=initializer_range,
                )
                if module.bias is not None:
                    module.bias.data.zero_()
        
        # Initialize layer norms
        for module in [block.norm1, block.norm2]:
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
        # Initialize layer scale parameters
        if hasattr(block, 'layer_scale1') and hasattr(block.layer_scale1, 'lambda1'):
            block.layer_scale1.lambda1.data.fill_(1e-5)
        if hasattr(block, 'layer_scale2') and hasattr(block.layer_scale2, 'lambda1'):
            block.layer_scale2.lambda1.data.fill_(1e-5)

    @add_start_docstrings_to_model_forward("Vision head forward pass.")
    def forward(
        self, 
        image_tokens: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the vision head.
        
        Args:
            image_tokens (`torch.Tensor`): 
                Input image tokens of shape (batch_size, sequence_length, hidden_size).
                
        Returns:
            `torch.Tensor`: Processed image tokens after attention blocks and linear projection.
        """
        # Process through attention blocks
        image_tokens = image_tokens.to(self.blocks[0].attention.k_proj.weight.dtype)
        for block in self.blocks:
            image_tokens = block(image_tokens)
            
        # Apply final layer normalization
        image_tokens = self.ln_final(image_tokens)
        
        # Apply linear projection
        return self.linear_projection(image_tokens)


@add_start_docstrings(VISION_TOWER_START_DOCSTRING)
class VisionTower(PreTrainedModel):
    """
    Vision tower module that combines a backbone with a vision head for processing images.
    """
    
    config_class = VisionTowerConfig
    config: VisionTowerConfig
    _no_split_modules = ["VisionHead", "SelfAttentionBlock"]
    _supports_flash_attn = True
    sub_configs = {"vision_head_config": VisionHeadConfig, "backbone_config": DINOv3ViTConfig}
    
    def __init__(self, config: VisionTowerConfig, backbone: Optional[nn.Module] = None):
        super().__init__(config)

        if backbone is None:
            assert config.backbone_config is not None, "backbone_config is required when backbone is not provided"
            # Create backbone config and ensure it inherits the attention implementation
            backbone_config_dict = config.backbone_config.copy()
            if hasattr(config, '_attn_implementation') and config._attn_implementation is not None:
                backbone_config_dict['_attn_implementation'] = config._attn_implementation
            self.backbone = DINOv3ViTModel(self.sub_configs["backbone_config"](**backbone_config_dict))
        
        assert self.backbone is not None, "backbone is not initialized"

        if self.backbone is not None and config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.use_class_token = config.use_class_token
        self.use_patch_tokens = config.use_patch_tokens
        self.patch_token_layer = config.patch_token_layer
        self.patch_tokens_pooler_type = config.patch_tokens_pooler_type
        
        # Get number of register tokens from backbone
        self.num_register_tokens = 0
        assert self.backbone is not None, "backbone is not initialized"
        if self.backbone is not None:
            if hasattr(self.backbone.config, "num_register_tokens"):
                self.num_register_tokens = self.backbone.config.num_register_tokens
            elif hasattr(self.backbone.config, "n_storage_tokens"):
                self.num_register_tokens = self.backbone.config.n_storage_tokens
            
            # Get backbone output dimension
            backbone_out_dim = getattr(self.backbone.config, 'hidden_size', 768) 
            num_heads = getattr(self.backbone.config, 'num_attention_heads', 12)
            logger.info(f"Visual backbone embedding dimension: {backbone_out_dim}")
            
            # Create vision head configuration
            head_config = dict(
                input_dim=backbone_out_dim,
                embed_dim=config.embed_dim,
                num_heads=num_heads,
                num_blocks=config.num_head_blocks,
                blocks_drop_path=config.head_blocks_block_drop_path,
                use_class_token=config.use_class_token,
                use_patch_tokens=config.use_patch_tokens,
                use_linear_projection=config.use_linear_projection,
            )
            
            # Pass through attention implementation if available
            if hasattr(config, '_attn_implementation') and config._attn_implementation is not None:
                head_config['_attn_implementation'] = config._attn_implementation
            
            # Initialize vision head
            self.head = VisionHead(self.sub_configs["vision_head_config"](**head_config))
        else:
            self.head = None
        
        assert self.head is not None, "head is not initialized"
        self.mlp_projector = nn.Sequential(
            nn.Linear(backbone_out_dim, config.language_embed_dim),
            nn.GELU(),
            nn.Linear(config.language_embed_dim, config.language_embed_dim),
        )
        # Initialize weights
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=module.in_features**-0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def init_weights(self):
        """Initialize weights for backbone and head."""
        super().init_weights()
        
        if self.backbone is not None and hasattr(self.backbone, 'init_weights'):
            self.backbone.init_weights()
        if self.head is not None:
            self.head.init_weights()

    def restore_image_grid_thw(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Restore image grid to original shape.
        """
        bs = image_grid_thw.shape[0]
        grid_t, grid_h, grid_w = image_grid_thw[0] # 所有图像分辨率都一样
        channel = 3
        temporal_patch_size = 1
        patch_size = 16

        patches = pixel_values.reshape(
            bs * grid_t * grid_h * grid_w, 
            channel, 
            temporal_patch_size, 
            patch_size, 
            patch_size
        )
        patches = patches.reshape(
            bs*grid_t,
            grid_h,
            grid_w,
            channel,
            temporal_patch_size,
            patch_size,
            patch_size
        )
    
        
        restored_pixel_values = patches.permute(0, 4, 3, 1, 5, 2, 6).contiguous()
        restored_pixel_values = restored_pixel_values.reshape(
                                    bs,
                                    grid_t * temporal_patch_size,
                                    channel,
                                    grid_h * patch_size,
                                    grid_w * patch_size
                                ).squeeze(1) # no temporal
        return restored_pixel_values
    
    def get_backbone_features(
        self, pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features from the backbone model.
        
        Args:
            pixel_values (`torch.Tensor`): Input pixel_values of shape (batch_size, channels, height, width).
            
        Returns:
            `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: 
                Tuple containing class token, patch tokens, and register tokens.
        """
        if self.backbone is None:
            raise ValueError("Backbone is not initialized")

        pixel_values = pixel_values.to(self.backbone.dtype)
        pixel_values = self.restore_image_grid_thw(pixel_values, image_grid_thw)
        tokens = self.backbone(
            pixel_values
        )
        class_token = tokens.last_hidden_state[:, 0]
        patch_tokens = tokens.last_hidden_state[:, self.num_register_tokens+1:]
        register_tokens = tokens.last_hidden_state[:, 1:self.num_register_tokens+1]
        return class_token, patch_tokens, register_tokens

    def get_class_and_patch_tokens(
        self, pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get processed class and patch tokens from pixel_values.
        
        Args:
            pixel_values (`torch.Tensor`): Input pixel_values of shape (batch_size, channels, height, width).
            
        Returns:
            `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: 
                Tuple containing processed class token, patch tokens, and backbone patch tokens.
        """
        class_token, patch_tokens, register_tokens = self.get_backbone_features(pixel_values, image_grid_thw)
        
        if self.head is None:
            raise ValueError("Vision head is not initialized")
            
        image_tokens = self.head(torch.cat([class_token.unsqueeze(1), register_tokens, patch_tokens], dim=1))
        
        return (
            image_tokens[:, 0],
            image_tokens[:, self.num_register_tokens + 1 :],
            patch_tokens,
        )

    @add_start_docstrings_to_model_forward("Vision tower forward pass.")
    def forward(
        self, 
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the vision tower.
        
        Args:
            pixel_values (`torch.Tensor`): 
                Input pixel_values of shape (batch_size, channels, height, width).
                
        Returns:
            `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: 
                Tuple containing pooled features, patch tokens, and backbone patch tokens.
        """
        class_token, patch_tokens, backbone_patch_tokens = (
            self.get_class_and_patch_tokens(pixel_values, image_grid_thw)
        )
        
        features = []
        
        if self.use_class_token:
            features.append(class_token)
            
        if self.use_patch_tokens:
            if self.patch_tokens_pooler_type == "mean":
                features.append(torch.mean(patch_tokens, dim=1))
            elif self.patch_tokens_pooler_type == "max":
                features.append(torch.max(patch_tokens, dim=1).values)
            else:
                raise ValueError(
                    f"Unknown patch tokens pooler type: {self.patch_tokens_pooler_type}"
                )
        patch_tokens = self.mlp_projector(patch_tokens)
        return torch.cat(features, dim=-1), patch_tokens, backbone_patch_tokens



@auto_docstring
class DINOv3ViTQwen2_5_VLPreTrainedModel(PreTrainedModel):
    config: DINOv3ViTQwen2_5_VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DINOv3ViTLayer", "Qwen2_5_VLDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True


class DINOv3ViTQwen2_5_VLModel(Qwen2_5_VLModel, DINOv3ViTQwen2_5_VLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    config: DINOv3ViTQwen2_5_VLConfig
    _no_split_modules = ["DINOv3ViTLayer", "Qwen2_5_VLDecoderLayer"]
    def __init__(self, config):
        DINOv3ViTQwen2_5_VLPreTrainedModel.__init__(self, config)
        config.vision_config.language_embed_dim = config.text_config.hidden_size
        self.visual = VisionTower._from_config(config.vision_config)
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)

        self.post_init()

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        pixel_values = pixel_values.to(self.visual.backbone.dtype)
        image_features = self.visual(pixel_values, image_grid_thw)[1]
        image_features = image_features.reshape(-1, image_features.shape[-1])
        split_sizes = image_grid_thw.prod(-1).tolist()
        image_features = torch.split(image_features, split_sizes)
        return image_features

class DINOv3ViTQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration, DINOv3ViTQwen2_5_VLPreTrainedModel):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        DINOv3ViTQwen2_5_VLPreTrainedModel.__init__(self, config)
        
        self.model = DINOv3ViTQwen2_5_VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()


__all__ = ["DINOv3ViTQwen2_5_VLForConditionalGeneration", "DINOv3ViTQwen2_5_VLModel", "DINOv3ViTQwen2_5_VLPreTrainedModel", "DINOv3ViTQwen2_5_VLTextModel"]