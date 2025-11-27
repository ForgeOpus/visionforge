"""TensorFlow Conv2D Layer Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, ConfigOption, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowNodeDefinition


class Conv2DNode(TensorFlowNodeDefinition):
    """
    2D Convolutional Layer.

    Uses tf.keras.layers.Conv2D.
    TensorFlow uses NHWC format by default.

    Input: [batch_size, height, width, channels]
    Output: [batch_size, new_height, new_width, filters]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="conv2d",
            label="Conv2D",
            category="basic",
            color="var(--color-purple)",
            icon="SquareHalf",
            description="2D convolutional layer",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="filters",
                label="Filters",
                type="number",
                required=True,
                min=1,
                description="Number of output filters"
            ),
            ConfigField(
                name="kernel_size",
                label="Kernel Size",
                type="number",
                default=3,
                min=1,
                description="Size of convolving kernel"
            ),
            ConfigField(
                name="strides",
                label="Strides",
                type="number",
                default=1,
                min=1,
                description="Stride of convolution"
            ),
            ConfigField(
                name="padding",
                label="Padding",
                type="select",
                default="valid",
                options=[
                    ConfigOption(value="valid", label="Valid"),
                    ConfigOption(value="same", label="Same")
                ],
                description="Padding mode"
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
                    ConfigOption(value="tanh", label="Tanh")
                ],
                description="Activation function"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        filters = config.get("filters")
        if not input_shape or filters is None or filters == '':
            return None

        # TensorFlow uses NHWC format
        if len(input_shape.dims) != 4:
            return None

        try:
            num_filters = int(filters)
            if num_filters <= 0:
                return None
        except (ValueError, TypeError):
            return None

        batch, height, width, _ = input_shape.dims
        kernel = int(config.get("kernel_size", 3))
        strides = int(config.get("strides", 1))
        padding = config.get("padding", "valid")

        # Calculate output dimensions
        if padding == "same":
            out_height = (height + strides - 1) // strides
            out_width = (width + strides - 1) // strides
        else:  # valid
            out_height = (height - kernel) // strides + 1
            out_width = (width - kernel) // strides + 1

        return TensorShape(
            dims=[batch, out_height, out_width, num_filters],
            description="Convolved feature map (NHWC)"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        if source_node_type in ("input", "dataloader", "empty", "custom"):
            return None

        return self.validate_dimensions(
            source_output_shape,
            4,
            "[batch, height, width, channels]"
        )
