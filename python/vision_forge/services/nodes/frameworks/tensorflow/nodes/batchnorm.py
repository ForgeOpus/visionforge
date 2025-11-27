"""TensorFlow BatchNormalization Layer Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowPassthroughNode


class BatchNormNode(TensorFlowPassthroughNode):
    """
    Batch Normalization Layer.

    Normalizes activations of previous layer.
    Uses tf.keras.layers.BatchNormalization.

    Input: any shape
    Output: same shape as input
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="batchnorm",
            label="BatchNorm",
            category="basic",
            color="var(--color-green)",
            icon="Normalize",
            description="Batch normalization",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="momentum",
                label="Momentum",
                type="number",
                default=0.99,
                min=0.0,
                max=1.0,
                description="Momentum for moving average"
            ),
            ConfigField(
                name="epsilon",
                label="Epsilon",
                type="number",
                default=0.001,
                description="Small constant for numerical stability"
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
            description="Normalized output"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return None  # Accepts any input
