"""TensorFlow Concatenate Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class ConcatNode(NodeDefinition):
    """Concatenate multiple tensors using tf.keras.layers.Concatenate"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="concat",
            label="Concat",
            category="merge",
            color="var(--color-accent)",
            icon="ArrowsMerge",
            description="Concatenate tensors",
            framework=Framework.TENSORFLOW
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="axis",
                label="Axis",
                type="number",
                default=-1,
                description="Axis along which to concatenate (use -1 for last axis, typically channels)"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # For concat, we need multiple inputs, but this method only sees one
        # The actual shape computation happens in the inference engine with all inputs
        # Here we just preserve the input structure
        if not input_shape:
            return None
        
        # Return same dimensions for now - actual concat dimension
        # will be computed by the inference engine with all inputs
        return TensorShape(
            dims=input_shape.dims,
            description="Concatenated (NHWC)"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Concat accepts multiple inputs - validation happens at the graph level
        # to ensure all inputs have compatible shapes
        return None
    
    def allows_multiple_inputs(self) -> bool:
        """Concat nodes accept multiple input connections"""
        return True
