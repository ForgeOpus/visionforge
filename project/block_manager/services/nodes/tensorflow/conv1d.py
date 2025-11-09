"""TensorFlow Conv1D Layer Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class Conv1DNode(NodeDefinition):
    """1D Convolutional Layer using tf.keras.layers.Conv1D"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="conv1d",
            label="Conv1D",
            category="basic",
            color="var(--color-purple)",
            icon="SquareHalf",
            description="1D convolutional layer (TensorFlow)",
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
                description="Number of output filters"
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
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or not config.get("filters"):
            return None
        
        # TensorFlow Conv1D expects: [batch, steps, channels]
        if len(input_shape.dims) != 3:
            return None
        
        batch, steps, _ = input_shape.dims
        kernel = int(config.get("kernel_size", 3))
        strides = int(config.get("strides", 1))
        padding = config.get("padding", "valid")
        
        # Calculate output dimensions
        if padding == "same":
            out_steps = (steps + strides - 1) // strides
        else:  # valid
            out_steps = (steps - kernel) // strides + 1
        
        return TensorShape(
            dims=[batch, out_steps, int(config["filters"])],
            description="1D convolved output"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        if source_node_type in ("input", "dataloader", "empty", "custom"):
            return None
        
        return self.validate_dimensions(
            source_output_shape,
            3,
            "[batch, steps, channels]"
        )
