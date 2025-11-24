"""TensorFlow Softmax Activation Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowPassthroughNode


class SoftmaxNode(TensorFlowPassthroughNode):
    """
    Softmax Activation Layer.

    Converts logits to probabilities.
    Uses tf.keras.layers.Softmax.

    Input: [batch_size, num_classes]
    Output: [batch_size, num_classes] (sums to 1)
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="softmax",
            label="Softmax",
            category="activation",
            color="var(--color-orange)",
            icon="Activity",
            description="Softmax activation",
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
                description="Axis to apply softmax"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape:
            return None

        return TensorShape(
            dims=input_shape.dims,
            description="Probability distribution"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return None  # Accepts any input
