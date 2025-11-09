"""TensorFlow GlobalAveragePooling2D Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class AdaptiveAvgPool2DNode(NodeDefinition):
    """Global Average Pooling using tf.keras.layers.GlobalAveragePooling2D"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="adaptiveavgpool2d",
            label="GlobalAvgPool2D",
            category="basic",
            color="var(--color-blue)",
            icon="Minimize",
            description="Global average pooling (adaptive)",
            framework=Framework.TENSORFLOW
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="keepdims",
                label="Keep Dimensions",
                type="boolean",
                default=False,
                description="Keep spatial dimensions (as 1x1)"
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
        
        batch, _, _, channels = input_shape.dims
        keepdims = config.get("keepdims", False)
        
        if keepdims:
            # Output: [batch, 1, 1, channels]
            return TensorShape(
                dims=[batch, 1, 1, channels],
                description="Global average pooled with dims"
            )
        else:
            # Output: [batch, channels]
            return TensorShape(
                dims=[batch, channels],
                description="Global average pooled"
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
