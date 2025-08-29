from typing import Optional

from transformers.configuration_utils import PretrainedConfig

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig


class VisionHeadConfig(PretrainedConfig):
    r"""
    Configuration class for VisionHead model.

    Args:
        input_dim (`int`, *optional*, defaults to 768):
            The input dimension of the vision head.
        embed_dim (`int`, *optional*, defaults to 768):
            The embedding dimension of the vision head.
        num_heads (`int`, *optional*, defaults to 12):
            The number of attention heads.
        num_blocks (`int`, *optional*, defaults to 2):
            The number of attention blocks.
        blocks_drop_path (`float`, *optional*, defaults to 0.1):
            The drop path rate for the attention blocks.
        use_class_token (`bool`, *optional*, defaults to True):
            Whether to use class token.
        use_patch_tokens (`bool`, *optional*, defaults to True):
            Whether to use patch tokens.
        use_linear_projection (`bool`, *optional*, defaults to False):
            Whether to use linear projection.
    """
    model_type = "vision_head"

    def __init__(
        self,
        input_dim: int = 768,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_blocks: int = 2,
        blocks_drop_path: float = 0.1,
        use_class_token: bool = True,
        use_patch_tokens: bool = True,
        use_linear_projection: bool = False,
        **kwargs,
    ):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.blocks_drop_path = blocks_drop_path
        self.use_class_token = use_class_token
        self.use_patch_tokens = use_patch_tokens
        self.use_linear_projection = use_linear_projection
        super().__init__(**kwargs)


class VisionTowerConfig(PretrainedConfig):
    r"""
    Configuration class for VisionTower model.

    Args:
        backbone_config (`dict`, *optional*):
            Configuration for the backbone model.
        freeze_backbone (`bool`, *optional*, defaults to False):
            Whether to freeze the backbone.
        embed_dim (`int`, *optional*, defaults to 768):
            The embedding dimension.
        num_head_blocks (`int`, *optional*, defaults to 2):
            The number of head blocks.
        head_blocks_block_drop_path (`float`, *optional*, defaults to 0.1):
            The drop path rate for head blocks.
        use_class_token (`bool`, *optional*, defaults to True):
            Whether to use class token.
        use_patch_tokens (`bool`, *optional*, defaults to True):
            Whether to use patch tokens.
        patch_token_layer (`int`, *optional*, defaults to -1):
            The layer to extract patch tokens from.
        patch_tokens_pooler_type (`str`, *optional*, defaults to "mean"):
            The pooling type for patch tokens.
        use_linear_projection (`bool`, *optional*, defaults to False):
            Whether to use linear projection.
    """
    model_type = "vision_tower"
    base_config_key = "vision_tower_config"

    def __init__(
        self,
        backbone_config: Optional[dict] = None,
        freeze_backbone: bool = False,
        embed_dim: int = 768,
        num_head_blocks: int = 2,
        head_blocks_block_drop_path: float = 0.1,
        use_class_token: bool = True,
        use_patch_tokens: bool = True,
        patch_token_layer: int = -1,
        patch_tokens_pooler_type: str = "mean",
        use_linear_projection: bool = False,
        spatial_merge_size: int = 1,
        temporal_patch_size: int = 1,
        tokens_per_second: int = 4,
        window_size: int = 112,
        out_hidden_size: int = 3584,
        fullatt_block_indexes: list[int] = [7, 15, 23, 31],
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.backbone_config = backbone_config
        self.freeze_backbone = freeze_backbone
        self.embed_dim = embed_dim
        self.num_head_blocks = num_head_blocks
        self.head_blocks_block_drop_path = head_blocks_block_drop_path
        self.use_class_token = use_class_token
        self.use_patch_tokens = use_patch_tokens
        self.patch_token_layer = patch_token_layer
        self.patch_tokens_pooler_type = patch_tokens_pooler_type
        self.use_linear_projection = use_linear_projection
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.out_hidden_size = out_hidden_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.initializer_range = initializer_range
        super().__init__(**kwargs)



class DINOv3ViTQwen2_5_VLConfig(Qwen2_5_VLConfig):
    model_type = "dinotxt_qwen2_5_vl"
    sub_configs = {"vision_config": VisionTowerConfig, "text_config": Qwen2_5_VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]