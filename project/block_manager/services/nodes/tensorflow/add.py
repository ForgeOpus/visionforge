"""TensorFlow Add Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class AddNode(NodeDefinition):
    """Element-wise addition using tf.keras.layers.Add"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="add",
            label="Add",
            category="merge",
            color="var(--color-accent)",
            icon="Plus",
            description="Element-wise addition",
            framework=Framework.TENSORFLOW
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return []  # Add operation doesn't need configuration
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Element-wise addition preserves shape
        # All inputs must have the same shape
        if not input_shape:
            return None
        
        return TensorShape(
            dims=input_shape.dims,
            description="Element-wise sum"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Add accepts multiple inputs - validation happens at graph level
        # to ensure all inputs have the same shape
        return None
    
    def allows_multiple_inputs(self) -> bool:
        """Add nodes accept multiple input connections"""
        return True
    def get_tensorflow_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate TensorFlow code specification for Add layer"""
        sanitized_id = node_id.replace('-', '_')
        class_name = 'AddBlock'
        layer_var = f'{sanitized_id}_AddBlock'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='add',
            node_id=node_id,
            init_params={},
            config_params=config,
            input_shape_info={},
            output_shape_info={},
            template_context={}
        )

