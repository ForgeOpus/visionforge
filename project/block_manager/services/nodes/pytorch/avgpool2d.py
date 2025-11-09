"""PyTorch AvgPool2D Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class AvgPool2DNode(NodeDefinition):
    """2D Average Pooling layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="avgpool2d",
            label="AvgPool2D",
            category="basic",
            color="var(--color-primary)",
            icon="ArrowsInSimple",
            description="Average pooling for 2D inputs",
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
        
        # Calculate output dimensions
        out_height = ((height + 2 * padding - kernel_size) // stride) + 1
        out_width = ((width + 2 * padding - kernel_size) // stride) + 1
        
        return TensorShape(
            dims=[batch, channels, out_height, out_width],
            description=f"AvgPool2D({kernel_size}x{kernel_size})"
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
