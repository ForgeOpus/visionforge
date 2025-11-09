"""TensorFlow Conv2D Layer Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class Conv2DNode(NodeDefinition):
    """2D Convolutional Layer using tf.keras.layers.Conv2D"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="conv2d",
            label="Conv2D",
            category="basic",
            color="var(--color-purple)",
            icon="SquareHalf",
            description="2D convolutional layer (TensorFlow)",
            framework=Framework.TENSORFLOW
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="filters",
                label="Filters",
                type="number",
                required=True,
                min=1,
                description="Number of output filters (channels)"
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
                name="strides",
                label="Strides",
                type="number",
                default=1,
                min=1,
                description="Stride of convolution"
            ),
            ConfigField(
                name="padding",
                label="Padding",
                type="select",
                default="valid",
                options=[
                    {"value": "valid", "label": "Valid (no padding)"},
                    {"value": "same", "label": "Same (preserve dimensions)"}
                ],
                description="Padding mode"
            ),
            ConfigField(
                name="activation",
                label="Activation",
                type="select",
                default="None",
                options=[
                    {"value": "None", "label": "None"},
                    {"value": "relu", "label": "ReLU"},
                    {"value": "sigmoid", "label": "Sigmoid"},
                    {"value": "tanh", "label": "Tanh"}
                ],
                description="Activation function"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or not config.get("filters"):
            return None
        
        # TensorFlow uses NHWC format: [batch, height, width, channels]
        if len(input_shape.dims) != 4:
            return None
        
        batch, height, width, _ = input_shape.dims
        kernel = int(config.get("kernel_size", 3))
        strides = int(config.get("strides", 1))
        padding = config.get("padding", "valid")
        
        # Calculate output dimensions
        if padding == "same":
            out_height = (height + strides - 1) // strides
            out_width = (width + strides - 1) // strides
        else:  # valid
            out_height = (height - kernel) // strides + 1
            out_width = (width - kernel) // strides + 1
        
        return TensorShape(
            dims=[batch, out_height, out_width, int(config["filters"])],
            description="Convolved feature map (NHWC)"
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
        
        # Validate dimension requirement (NHWC format)
        return self.validate_dimensions(
            source_output_shape,
            4,
            "[batch, height, width, channels]"
        )
