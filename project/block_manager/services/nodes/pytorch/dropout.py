"""PyTorch Dropout Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class DropoutNode(NodeDefinition):
    """Dropout regularization layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="dropout",
            label="Dropout",
            category="basic",
            color="var(--color-primary)",
            icon="Percent",
            description="Dropout regularization",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="p",
                label="Dropout Rate",
                type="number",
                default=0.5,
                min=0.0,
                max=1.0,
                description="Probability of dropping a unit (0 to 1)"
            ),
            ConfigField(
                name="inplace",
                label="In-place",
                type="boolean",
                default=False,
                description="Perform operation in-place"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Dropout preserves shape
        if input_shape:
            return TensorShape(
                dims=input_shape.dims,
                description=f"Dropout ({config.get('p', 0.5)})"
            )
        return None
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Dropout accepts any input shape
        return None

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate PyTorch code specification for Dropout layer"""
        p = config.get('p', 0.5)

        sanitized_id = node_id.replace('-', '_')
        class_name = 'DropoutLayer'
        layer_var = f'{sanitized_id}_DropoutLayer'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='dropout',
            node_id=node_id,
            init_params={'p': p},
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={'p': p}
        )
