"""PyTorch BatchNorm2D Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class BatchNorm2DNode(NodeDefinition):
    """2D Batch Normalization layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="batchnorm2d",
            label="BatchNorm2D",
            category="basic",
            color="var(--color-primary)",
            icon="ChartLineUp",
            description="Batch normalization for 2D inputs",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="num_features",
                label="Number of Features",
                type="number",
                required=True,
                min=1,
                description="Number of channels (C from [N, C, H, W])"
            ),
            ConfigField(
                name="eps",
                label="Epsilon",
                type="number",
                default=1e-5,
                description="Value for numerical stability"
            ),
            ConfigField(
                name="momentum",
                label="Momentum",
                type="number",
                default=0.1,
                min=0.0,
                max=1.0,
                description="Momentum for running mean/variance"
            ),
            ConfigField(
                name="affine",
                label="Affine",
                type="boolean",
                default=True,
                description="Use learnable affine parameters"
            ),
            ConfigField(
                name="track_running_stats",
                label="Track Stats",
                type="boolean",
                default=True,
                description="Track running mean and variance"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # BatchNorm2D preserves shape
        if input_shape:
            return TensorShape(
                dims=input_shape.dims,
                description="Batch normalized"
            )
        return None
    
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
        
        # Validate 4D input (N, C, H, W)
        return self.validate_dimensions(
            source_output_shape,
            4,
            "[batch, channels, height, width]"
        )
