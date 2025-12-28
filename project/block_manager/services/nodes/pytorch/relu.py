"""PyTorch ReLU Activation Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class ReLUNode(NodeDefinition):
    """ReLU activation function layer"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="relu",
            label="ReLU",
            category="basic",
            color="var(--color-primary)",
            icon="Zap",
            description="ReLU activation function",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
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
        # ReLU preserves shape
        if input_shape:
            return TensorShape(
                dims=input_shape.dims,
                description="ReLU activated"
            )
        return None

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # ReLU accepts any input shape
        return None

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate PyTorch code specification for ReLU layer"""
        inplace = config.get('inplace', False)

        sanitized_id = node_id.replace('-', '_')
        class_name = 'ReLULayer'
        layer_var = f'{sanitized_id}_ReLULayer'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='relu',
            node_id=node_id,
            init_params={'inplace': inplace},
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={'inplace': inplace}
        )
