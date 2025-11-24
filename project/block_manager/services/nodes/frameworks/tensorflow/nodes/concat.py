"""TensorFlow Concatenate Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowMergeNode


class ConcatNode(TensorFlowMergeNode):
    """
    Concatenation Layer.

    Concatenates multiple inputs along specified axis.
    Uses tf.keras.layers.Concatenate.

    Input: List of tensors with compatible shapes
    Output: Concatenated tensor
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="concat",
            label="Concatenate",
            category="merge",
            color="var(--color-yellow)",
            icon="GitMerge",
            description="Concatenate tensors",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="axis",
                label="Axis",
                type="number",
                default=-1,
                description="Axis to concatenate along"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # For single input, return as-is
        if not input_shape:
            return None

        return TensorShape(
            dims=input_shape.dims,
            description="Concatenated output"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Accept multiple inputs
        return None
