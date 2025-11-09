"""TensorFlow Flatten Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class FlattenNode(NodeDefinition):
    """Flatten layer using tf.keras.layers.Flatten"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="flatten",
            label="Flatten",
            category="basic",
            color="var(--color-primary)",
            icon="Rows",
            description="Flatten tensor to 2D",
            framework=Framework.TENSORFLOW
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return []  # Flatten doesn't need configuration in TensorFlow
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) < 2:
            return None
        
        batch_size = input_shape.dims[0]
        
        # Calculate flattened features (multiply all dimensions after batch)
        features = 1
        for dim in input_shape.dims[1:]:
            features *= dim
        
        return TensorShape(
            dims=[batch_size, features],
            description="Flattened tensor"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Allow connections from input/dataloader without shape validation
        if source_node_type in ("input", "dataloader", "empty", "custom"):
            return None
        
        # Validate that input has at least 2 dimensions
        if source_output_shape and len(source_output_shape.dims) < 2:
            return "Flatten requires input with at least 2 dimensions"
        
        return None
