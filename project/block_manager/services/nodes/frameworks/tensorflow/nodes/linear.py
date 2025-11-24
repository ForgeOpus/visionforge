"""TensorFlow Dense (Linear) Layer Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, ConfigOption, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowNodeDefinition


class LinearNode(TensorFlowNodeDefinition):
    """
    Dense/Fully Connected Layer.

    Applies a linear transformation: y = xW + b
    Uses tf.keras.layers.Dense.

    Input: [batch_size, in_features] or [batch_size, ..., in_features]
    Output: [batch_size, units] or [batch_size, ..., units]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="linear",
            label="Dense",
            category="basic",
            color="var(--color-primary)",
            icon="Lightning",
            description="Fully connected layer (Dense)",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="units",
                label="Units",
                type="number",
                required=True,
                min=1,
                description="Number of output units"
            ),
            ConfigField(
                name="activation",
                label="Activation",
                type="select",
                default="None",
                options=[
                    ConfigOption(value="None", label="None"),
                    ConfigOption(value="relu", label="ReLU"),
                    ConfigOption(value="sigmoid", label="Sigmoid"),
                    ConfigOption(value="tanh", label="Tanh"),
                    ConfigOption(value="softmax", label="Softmax")
                ],
                description="Activation function"
            ),
            ConfigField(
                name="use_bias",
                label="Use Bias",
                type="boolean",
                default=True,
                description="Add learnable bias"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        units = config.get("units")
        if not input_shape or units is None or units == '':
            return None

        if len(input_shape.dims) < 2:
            return None

        try:
            num_units = int(units)
            if num_units <= 0:
                return None
        except (ValueError, TypeError):
            return None

        # Dense layer outputs [batch, ..., units]
        output_dims = input_shape.dims[:-1] + [num_units]

        return TensorShape(
            dims=output_dims,
            description="Dense layer output"
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
            return "Dense layer requires at least 2D input [batch, features, ...]"

        return None
