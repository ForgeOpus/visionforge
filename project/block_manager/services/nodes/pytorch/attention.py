"""PyTorch Multi-Head Attention Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class AttentionNode(NodeDefinition):
    """Multi-Head Attention Layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="attention",
            label="Multi-Head Attention",
            category="advanced",
            color="#8b5cf6",
            icon="Brain",
            description="Multi-head self-attention mechanism",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="embed_dim",
                label="Embedding Dimension",
                type="number",
                required=True,
                min=1,
                description="Total dimension of the model"
            ),
            ConfigField(
                name="num_heads",
                label="Number of Heads",
                type="number",
                required=True,
                min=1,
                description="Number of parallel attention heads"
            ),
            ConfigField(
                name="dropout",
                label="Dropout",
                type="number",
                default=0.0,
                min=0.0,
                max=1.0,
                description="Dropout probability on attention weights"
            ),
            ConfigField(
                name="bias",
                label="Use Bias",
                type="boolean",
                default=True,
                description="Add bias to input/output projection layers"
            ),
            ConfigField(
                name="add_bias_kv",
                label="Add Bias to K/V",
                type="boolean",
                default=False,
                description="Add bias to the key and value sequences"
            ),
            ConfigField(
                name="batch_first",
                label="Batch First",
                type="boolean",
                default=True,
                description="If True, input/output shape is (batch, seq, feature)"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or not config.get("embed_dim"):
            return None
        
        # Attention preserves the sequence length and uses embed_dim
        if len(input_shape.dims) != 3:
            return None
        
        batch_first = config.get("batch_first", True)
        embed_dim = int(config["embed_dim"])
        
        if batch_first:
            # (batch, seq, embed_dim)
            return TensorShape(
                dims=[input_shape.dims[0], input_shape.dims[1], embed_dim],
                description="Multi-head attention output"
            )
        else:
            # (seq, batch, embed_dim)
            return TensorShape(
                dims=[input_shape.dims[0], input_shape.dims[1], embed_dim],
                description="Multi-head attention output"
            )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Allow connections from input/dataloader without shape validation
        if source_node_type in ("input", "dataloader"):
            return None
        
        # Empty and custom nodes are flexible
        if source_node_type in ("empty", "custom"):
            return None
        
        # Validate 3D input requirement
        error = self.validate_dimensions(
            source_output_shape,
            3,
            "[batch, sequence, features] or [sequence, batch, features]"
        )
        
        if error:
            return error
        
        # Validate embed_dim divisibility by num_heads
        if target_config.get("embed_dim") and target_config.get("num_heads"):
            embed_dim = int(target_config["embed_dim"])
            num_heads = int(target_config["num_heads"])
            
            if embed_dim % num_heads != 0:
                return f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        return None
