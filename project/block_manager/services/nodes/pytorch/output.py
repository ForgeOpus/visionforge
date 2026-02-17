"""PyTorch Output Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class OutputNode(NodeDefinition):
    """Output node for defining model output and predictions"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="output",
            label="Output",
            category="output",
            color="var(--color-green)",
            icon="Export",
            description="Define model output and predictions",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return []  # No configuration needed

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Output node passes through the input shape
        return input_shape

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Output node accepts any input shape (final layer predictions)
        return None

    @property
    def allows_multiple_inputs(self) -> bool:
        """Output nodes accept single input"""
        return False

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """
        Output nodes don't generate layer code - they only mark the end of the model.
        This method exists for interface compatibility.
        """
        sanitized_id = node_id.replace('-', '_')

        return LayerCodeSpec(
            class_name='Output',
            layer_variable_name=f'{sanitized_id}_Output',
            node_type='output',
            node_id=node_id,
            init_params={},
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={}
        )
