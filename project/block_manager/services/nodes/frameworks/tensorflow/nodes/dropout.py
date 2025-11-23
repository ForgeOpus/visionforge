"""TensorFlow Dropout Layer Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowPassthroughNode


class DropoutNode(TensorFlowPassthroughNode):
    """
    Dropout Regularization Layer.

    Randomly drops units during training.
    Uses tf.keras.layers.Dropout.

    Input: any shape
    Output: same shape as input
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="dropout",
            label="Dropout",
            category="basic",
            color="var(--color-gray)",
            icon="CircleSlash",
            description="Dropout regularization",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="rate",
                label="Dropout Rate",
                type="number",
                default=0.5,
                min=0.0,
                max=1.0,
                description="Fraction of inputs to drop"
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
            description="Dropout applied"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return None  # Accepts any input
