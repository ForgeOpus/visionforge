"""PyTorch Concat Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class ConcatNode(NodeDefinition):
    """Concatenate multiple tensors along a dimension"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="concat",
            label="Concat",
            category="merge",
            color="var(--color-accent)",
            icon="ArrowsMerge",
            description="Concatenate tensors",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="dim",
                label="Dimension",
                type="number",
                default=1,
                description="Dimension along which to concatenate (typically 1 for channel dimension)"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # For concat, we need multiple inputs, but this method only sees one
        # The actual shape computation happens in the frontend/backend coordination
        # Here we just preserve the input structure
        if not input_shape:
            return None
        
        # Return same dimensions for now - actual concat dimension
        # will be computed by the inference engine with all inputs
        return TensorShape(
            dims=input_shape.dims,
            description="Concatenated"
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
    
    @property
    def allows_multiple_inputs(self) -> bool:
        """Concat nodes accept multiple input connections"""
        return True

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate PyTorch code specification for Concat layer"""
        dim = config.get('dim', 1)

        sanitized_id = node_id.replace('-', '_')
        class_name = 'ConcatBlock'
        layer_var = f'{sanitized_id}_ConcatBlock'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='concat',
            node_id=node_id,
            init_params={},
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={'dim': dim}
        )
