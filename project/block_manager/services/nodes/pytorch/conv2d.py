"""PyTorch Conv2D Layer Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class Conv2DNode(NodeDefinition):
    """2D Convolutional Layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="conv2d",
            label="Conv2D",
            category="basic",
            color="var(--color-purple)",
            icon="SquareHalf",
            description="2D convolutional layer",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="out_channels",
                label="Output Channels",
                type="number",
                required=True,
                min=1,
                description="Number of output channels"
            ),
            ConfigField(
                name="kernel_size",
                label="Kernel Size",
                type="number",
                default=3,
                min=1,
                description="Size of convolving kernel"
            ),
            ConfigField(
                name="stride",
                label="Stride",
                type="number",
                default=1,
                min=1,
                description="Stride of convolution"
            ),
            ConfigField(
                name="padding",
                label="Padding",
                type="number",
                default=0,
                min=0,
                description="Zero-padding added to both sides"
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
        if not input_shape or not config.get("out_channels"):
            return None
        
        if len(input_shape.dims) != 4:
            return None
        
        batch, _, height, width = input_shape.dims
        kernel = int(config.get("kernel_size", 3))
        stride = int(config.get("stride", 1))
        padding = int(config.get("padding", 0))
        dilation = int(config.get("dilation", 1))
        
        out_height, out_width = self.compute_conv2d_output(
            height, width, kernel, stride, padding, dilation
        )
        
        return TensorShape(
            dims=[batch, int(config["out_channels"]), out_height, out_width],
            description="Convolved feature map"
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
        
        # Validate dimension requirement
        return self.validate_dimensions(
            source_output_shape,
            4,
            "[batch, channels, height, width]"
        )
