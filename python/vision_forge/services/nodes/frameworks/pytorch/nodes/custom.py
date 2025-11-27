"""PyTorch Custom Layer Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchNodeDefinition


class CustomNode(PyTorchNodeDefinition):
    """
    Custom user-defined layer.

    Allows users to define custom forward pass logic.
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="custom",
            label="Custom Layer",
            category="advanced",
            color="var(--color-purple)",
            icon="Code",
            description="Custom user-defined layer",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="name",
                label="Layer Name",
                type="text",
                required=True,
                description="Name of the custom layer"
            ),
            ConfigField(
                name="code",
                label="Python Code",
                type="text",
                required=True,
                description="Custom forward pass implementation"
            ),
            ConfigField(
                name="output_shape",
                label="Output Shape",
                type="text",
                description="Expected output shape (optional)"
            ),
            ConfigField(
                name="description",
                label="Description",
                type="text",
                description="Brief description of the layer functionality"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Try to parse user-provided output shape
        output_shape_str = config.get("output_shape", "")
        if output_shape_str:
            dims = self.parse_shape_string(output_shape_str)
            if dims:
                return TensorShape(
                    dims=dims,
                    description="Custom layer output"
                )
        # If no output shape provided, pass through input shape
        return input_shape

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return None  # Accept any input
