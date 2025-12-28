"""PyTorch Softmax Activation Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class SoftmaxNode(NodeDefinition):
    """Softmax activation function layer"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="softmax",
            label="Softmax",
            category="basic",
            color="var(--color-primary)",
            icon="Activity",
            description="Softmax activation function",
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
                description="Dimension along which softmax will be computed"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Softmax preserves shape
        if input_shape:
            return TensorShape(
                dims=input_shape.dims,
                description="Softmax probabilities"
            )
        return None

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Softmax accepts any input shape
        return None

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate PyTorch code specification for Softmax layer"""
        dim = config.get('dim', 1)

        sanitized_id = node_id.replace('-', '_')
        class_name = 'SoftmaxLayer'
        layer_var = f'{sanitized_id}_SoftmaxLayer'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='softmax',
            node_id=node_id,
            init_params={'dim': dim},
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={'dim': dim}
        )
