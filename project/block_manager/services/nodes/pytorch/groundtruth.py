"""PyTorch Ground Truth Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class GroundTruthNode(NodeDefinition):
    """Ground truth labels for training"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="groundtruth",
            label="Ground Truth",
            category="input",
            color="var(--color-orange)",
            icon="Target",
            description="Ground truth labels for training",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="shape",
                label="Label Shape",
                type="string",
                default="[1, 10]",
                description="Ground truth tensor dimensions as JSON array (e.g., [batch, num_classes])"
            ),
            ConfigField(
                name="label",
                label="Custom Label",
                type="string",
                default="Ground Truth",
                description="Custom label for this ground truth node"
            ),
            ConfigField(
                name="note",
                label="Note",
                type="string",
                default="",
                description="Notes or comments about this ground truth data"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Parse shape from config
        shape_str = config.get("shape", "[1, 10]")
        dims = self.parse_shape_string(shape_str)

        if dims:
            return TensorShape(
                dims=dims,
                description="Ground truth labels"
            )

        # Fallback
        return TensorShape(
            dims=[1, 10],
            description="Ground truth labels"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Ground truth is a source node, doesn't accept incoming connections
        return "Ground Truth is a source node and cannot accept incoming connections"

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """
        Ground truth nodes don't generate layer code - they only provide data
        for the training script. This method exists for interface compatibility.
        """
        sanitized_id = node_id.replace('-', '_')

        return LayerCodeSpec(
            class_name='GroundTruth',
            layer_variable_name=f'{sanitized_id}_GroundTruth',
            node_type='groundtruth',
            node_id=node_id,
            init_params={},
            config_params=config,
            input_shape_info={'dims': []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={}
        )
