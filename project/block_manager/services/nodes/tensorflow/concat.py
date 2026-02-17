"""TensorFlow Concatenate Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


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
        # Concat accepts multiple inputs
        # Individual connection validation is basic - full multi-input validation
        # happens at graph level to ensure all inputs have compatible shapes
        # (same number of dimensions, matching sizes except on concat axis)

        # Ensure source provides a valid output shape
        if not source_output_shape or not source_output_shape.dims:
            return "Concat node requires inputs with defined shapes"

        # Validate concat axis is valid for input shape
        concat_axis = int(target_config.get('axis', -1))
        ndim = len(source_output_shape.dims)

        # Normalize negative axis
        if concat_axis < 0:
            concat_axis = ndim + concat_axis

        if concat_axis < 0 or concat_axis >= ndim:
            return f"Concat axis {target_config.get('axis', -1)} is invalid for {ndim}D tensor"

        return None
    
    def allows_multiple_inputs(self) -> bool:
        """Concat nodes accept multiple input connections"""
        return True
    def get_tensorflow_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate TensorFlow code specification for Concatenate layer"""
        axis = config.get('axis', -1)

        sanitized_id = node_id.replace('-', '_')
        class_name = 'ConcatBlock'
        layer_var = f'{sanitized_id}_ConcatBlock'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='concat',
            node_id=node_id,
            init_params={'axis': axis},
            config_params=config,
            input_shape_info={},
            output_shape_info={},
            template_context={'axis': axis}
        )

