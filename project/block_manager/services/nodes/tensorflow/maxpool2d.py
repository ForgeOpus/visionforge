"""TensorFlow MaxPooling2D Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class MaxPool2DNode(NodeDefinition):
    """Max Pooling Layer using tf.keras.layers.MaxPooling2D"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="maxpool2d",
            label="MaxPool2D",
            category="basic",
            color="var(--color-blue)",
            icon="ArrowDown",
            description="Max pooling layer",
            framework=Framework.TENSORFLOW
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="pool_size",
                label="Pool Size",
                type="number",
                default=2,
                min=1,
                description="Size of pooling window"
            ),
            ConfigField(
                name="strides",
                label="Strides",
                type="number",
                default=2,
                min=1,
                description="Stride of pooling operation"
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
        if not input_shape:
            return None
        
        # NHWC format: [batch, height, width, channels]
        if len(input_shape.dims) != 4:
            return None
        
        batch, height, width, channels = input_shape.dims
        pool_size = int(config.get("pool_size", 2))
        strides = int(config.get("strides", 2))
        padding = config.get("padding", "valid")
        
        # Calculate output dimensions
        if padding == "same":
            out_height = (height + strides - 1) // strides
            out_width = (width + strides - 1) // strides
        else:  # valid
            out_height = (height - pool_size) // strides + 1
            out_width = (width - pool_size) // strides + 1
        
        return TensorShape(
            dims=[batch, out_height, out_width, channels],
            description="Max pooled (NHWC)"
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
            4,
            "[batch, height, width, channels]"
        )
