"""PyTorch Ground Truth Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


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
