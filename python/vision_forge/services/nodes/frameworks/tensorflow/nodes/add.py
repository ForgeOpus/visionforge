"""TensorFlow Add Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowMergeNode


class AddNode(TensorFlowMergeNode):
    """
    Element-wise Addition Layer.

    Adds multiple inputs element-wise.
    Uses tf.keras.layers.Add.

    Input: List of tensors with identical shapes
    Output: Sum of inputs
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="add",
            label="Add",
            category="merge",
            color="var(--color-yellow)",
            icon="Plus",
            description="Element-wise addition",
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
        if not input_shape:
            return None

        return TensorShape(
            dims=input_shape.dims,
            description="Sum of inputs"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Accept multiple inputs
        return None
