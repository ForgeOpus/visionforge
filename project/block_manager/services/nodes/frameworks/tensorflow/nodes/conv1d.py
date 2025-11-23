"""TensorFlow Conv1D Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowNodeDefinition


class Conv1DNode(TensorFlowNodeDefinition):
    """
    1D Convolutional Layer (Keras).

    Input: [batch_size, length, channels]
    Output: [batch_size, output_length, filters]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="conv1d",
            label="Conv1D",
            category="basic",
            color="var(--color-purple)",
            icon="SquareHalf",
            description="1D convolutional layer",
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
                    {"value": "valid", "label": "Valid"},
                    {"value": "same", "label": "Same"},
                ],
                description="Padding mode"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 3:
            return None

        batch, length, _ = input_shape.dims
        filters = config.get("filters")
        if filters is None:
            return None

        kernel_size = int(config.get("kernel_size", 3))
        strides = int(config.get("strides", 1))
        padding = config.get("padding", "valid")

        if padding == "same":
            out_length = (length + strides - 1) // strides
        else:
            out_length = (length - kernel_size) // strides + 1

        return TensorShape(
            dims=[batch, out_length, int(filters)],
            description="1D convolved output"
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
            3,
            "[batch, length, channels]"
        )
