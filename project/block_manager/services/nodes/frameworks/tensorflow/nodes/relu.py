"""TensorFlow ReLU Activation Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowPassthroughNode


class ReLUNode(TensorFlowPassthroughNode):
    """
    ReLU Activation Layer.

    Applies rectified linear unit activation.
    Uses tf.keras.layers.ReLU.

    Input: any shape
    Output: same shape as input
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="relu",
            label="ReLU",
            category="activation",
            color="var(--color-orange)",
            icon="Activity",
            description="ReLU activation function",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="max_value",
                label="Max Value",
                type="number",
                default=None,
                description="Maximum value for output (None = no limit)"
            ),
            ConfigField(
                name="negative_slope",
                label="Negative Slope",
                type="number",
                default=0.0,
                description="Slope for negative values (Leaky ReLU)"
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
            description="ReLU activated"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return None  # Accepts any input
