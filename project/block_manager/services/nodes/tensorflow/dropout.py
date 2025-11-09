"""TensorFlow Dropout Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class DropoutNode(NodeDefinition):
    """Dropout Layer using tf.keras.layers.Dropout"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="dropout",
            label="Dropout",
            category="basic",
            color="var(--color-gray)",
            icon="CircleSlash",
            description="Dropout regularization",
            framework=Framework.TENSORFLOW
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="rate",
                label="Dropout Rate",
                type="number",
                default=0.5,
                min=0.0,
                max=1.0,
                description="Fraction of inputs to drop (0.0 - 1.0)"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Dropout preserves input shape
        if not input_shape:
            return None
        
        return TensorShape(
            dims=input_shape.dims,
            description="Dropout applied"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Dropout accepts any input
        return None
