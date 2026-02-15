"""PyTorch Conv3D Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class Conv3DNode(NodeDefinition):
    """3D Convolution layer for volumetric data"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="conv3d",
            label="Conv3D",
            category="advanced",
            color="var(--color-purple)",
            icon="Cube",
            description="3D convolutional layer",
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
                description="Zero padding on all sides"
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
        if not input_shape or len(input_shape.dims) != 5:
            return None
        
        batch, in_channels, depth, height, width = input_shape.dims
        out_channels = int(config.get("out_channels", in_channels))
        kernel_size = int(config.get("kernel_size", 3))
        stride = int(config.get("stride", 1))
        padding = int(config.get("padding", 0))
        dilation = int(config.get("dilation", 1))
        
        # Calculate output dimensions
        out_depth = ((depth + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        out_height = ((height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        out_width = ((width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        
        return TensorShape(
            dims=[batch, out_channels, out_depth, out_height, out_width],
            description=f"Conv3D({out_channels})"
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

        # Validate 5D input (N, C, D, H, W)
        return self.validate_dimensions(
            source_output_shape,
            5,
            "[batch, channels, depth, height, width]"
        )

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate PyTorch code specification for Conv3D layer"""
        out_channels = config.get('out_channels', 64)
        kernel_size = config.get('kernel_size', 3)
        stride = config.get('stride', 1)
        padding = config.get('padding', 0)
        dilation = config.get('dilation', 1)
        bias = config.get('bias', True)

        # Determine in_channels from input shape if available
        in_channels = None
        if input_shape and len(input_shape.dims) >= 2:
            in_channels = input_shape.dims[1]

        sanitized_id = node_id.replace('-', '_')
        class_name = 'Conv3DBlock'
        layer_var = f'{sanitized_id}_Conv3DBlock'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='conv3d',
            node_id=node_id,
            init_params={
                'in_channels': in_channels,
                'out_channels': out_channels,
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'bias': bias
            },
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={
                'in_channels': in_channels,
                'out_channels': out_channels,
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'bias': bias
            }
        )
