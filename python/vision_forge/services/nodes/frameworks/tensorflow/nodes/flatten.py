"""TensorFlow Flatten Layer Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowNodeDefinition


class FlattenNode(TensorFlowNodeDefinition):
    """
    Flatten Layer.

    Flattens input to 2D tensor.
    Uses tf.keras.layers.Flatten.

    Input: [batch_size, ...]
    Output: [batch_size, flattened_features]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="flatten",
            label="Flatten",
            category="basic",
            color="var(--color-primary)",
            icon="Rows",
            description="Flatten tensor to 2D",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return []  # No config needed

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) < 2:
            return None

        batch_size = input_shape.dims[0]
        features = 1
        for dim in input_shape.dims[1:]:
            features *= dim

        return TensorShape(
            dims=[batch_size, features],
            description="Flattened tensor"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        if source_node_type in ("input", "dataloader", "empty", "custom"):
            return None

        if source_output_shape and len(source_output_shape.dims) < 2:
            return "Flatten requires at least 2D input"

        return None
