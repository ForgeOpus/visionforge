"""PyTorch MaxPool2D Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class MaxPool2DNode(NodeDefinition):
    """2D Max Pooling layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="maxpool2d",
            label="MaxPool2D",
            category="basic",
            color="var(--color-primary)",
            icon="ArrowsInSimple",
            description="Max pooling for 2D inputs",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="kernel_size",
                label="Kernel Size",
                type="number",
                default=2,
                min=1,
                description="Size of the pooling window"
            ),
            ConfigField(
                name="stride",
                label="Stride",
                type="number",
                default=2,
                min=1,
                description="Stride of the pooling window"
            ),
            ConfigField(
                name="padding",
                label="Padding",
                type="number",
                default=0,
                min=0,
                description="Zero padding on both sides"
            ),
            ConfigField(
                name="dilation",
                label="Dilation",
                type="number",
                default=1,
                min=1,
                description="Spacing between kernel elements"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 4:
            return None
        
        batch, channels, height, width = input_shape.dims
        kernel_size = int(config.get("kernel_size", 2))
        stride = int(config.get("stride", 2))
        padding = int(config.get("padding", 0))
        dilation = int(config.get("dilation", 1))
        
        # Calculate output dimensions
        out_height = ((height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        out_width = ((width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        
        return TensorShape(
            dims=[batch, channels, out_height, out_width],
            description=f"MaxPool2D({kernel_size}x{kernel_size})"
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
        
        # Validate 4D input (N, C, H, W)
        return self.validate_dimensions(
            source_output_shape,
            4,
            "[batch, channels, height, width]"
        )
