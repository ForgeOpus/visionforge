"""PyTorch Conv1D Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class Conv1DNode(NodeDefinition):
    """1D Convolution layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="conv1d",
            label="Conv1D",
            category="advanced",
            color="var(--color-purple)",
            icon="WaveSquare",
            description="1D convolutional layer",
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
                description="Size of the convolving kernel"
            ),
            ConfigField(
                name="stride",
                label="Stride",
                type="number",
                default=1,
                min=1,
                description="Stride of the convolution"
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
            ),
            ConfigField(
                name="bias",
                label="Use Bias",
                type="boolean",
                default=True,
                description="Add learnable bias"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 3:
            return None
        
        batch, in_channels, length = input_shape.dims
        out_channels = int(config.get("out_channels", in_channels))
        kernel_size = int(config.get("kernel_size", 3))
        stride = int(config.get("stride", 1))
        padding = int(config.get("padding", 0))
        dilation = int(config.get("dilation", 1))
        
        # Calculate output length
        out_length = ((length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        
        return TensorShape(
            dims=[batch, out_channels, out_length],
            description=f"Conv1D({out_channels})"
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
        
        # Validate 3D input (N, C, L)
        return self.validate_dimensions(
            source_output_shape,
            3,
            "[batch, channels, length]"
        )
