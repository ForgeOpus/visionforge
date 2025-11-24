"""PyTorch Flatten Layer Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchNodeDefinition


class FlattenNode(PyTorchNodeDefinition):
    """Flatten multi-dimensional input to 2D"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="flatten",
            label="Flatten",
            category="basic",
            color="var(--color-primary)",
            icon="ListBullets",
            description="Flatten multi-dimensional input to 2D",
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
        if not input_shape:
            return None

        if len(input_shape.dims) < 2:
            return None

        # Calculate flattened size (multiply all dims except batch)
        batch = input_shape.dims[0]
        flat_size = 1
        for dim in input_shape.dims[1:]:
            flat_size *= dim

        return TensorShape(
            dims=[batch, flat_size],
            description="Flattened features"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return None  # Accept any input
