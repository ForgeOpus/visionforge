"""TensorFlow MaxPool2D Layer Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, ConfigOption, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowNodeDefinition


class MaxPool2DNode(TensorFlowNodeDefinition):
    """
    2D Max Pooling Layer.

    Downsamples input using maximum values.
    Uses tf.keras.layers.MaxPool2D.

    Input: [batch_size, height, width, channels]
    Output: [batch_size, new_height, new_width, channels]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="maxpool2d",
            label="MaxPool2D",
            category="basic",
            color="var(--color-blue)",
            icon="ArrowDown",
            description="2D max pooling",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="pool_size",
                label="Pool Size",
                type="number",
                default=2,
                min=1,
                description="Size of pooling window"
            ),
            ConfigField(
                name="strides",
                label="Strides",
                type="number",
                default=2,
                min=1,
                description="Stride of pooling"
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
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 4:
            return None

        batch, height, width, channels = input_shape.dims
        pool_size = int(config.get("pool_size", 2))
        strides = int(config.get("strides", 2))
        padding = config.get("padding", "valid")

        if padding == "same":
            out_height = (height + strides - 1) // strides
            out_width = (width + strides - 1) // strides
        else:
            out_height = (height - pool_size) // strides + 1
            out_width = (width - pool_size) // strides + 1

        return TensorShape(
            dims=[batch, out_height, out_width, channels],
            description="Pooled output (NHWC)"
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
