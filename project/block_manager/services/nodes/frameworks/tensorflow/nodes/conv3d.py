"""TensorFlow Conv3D Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowNodeDefinition


class Conv3DNode(TensorFlowNodeDefinition):
    """
    3D Convolutional Layer (Keras).

    Input: [batch_size, depth, height, width, channels] (NDHWC)
    Output: [batch_size, out_depth, out_height, out_width, filters]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="conv3d",
            label="Conv3D",
            category="basic",
            color="var(--color-purple)",
            icon="Cube",
            description="3D convolutional layer",
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
        if not input_shape or len(input_shape.dims) != 5:
            return None

        batch, depth, height, width, _ = input_shape.dims
        filters = config.get("filters")
        if filters is None:
            return None

        kernel_size = int(config.get("kernel_size", 3))
        strides = int(config.get("strides", 1))
        padding = config.get("padding", "valid")

        if padding == "same":
            out_depth = (depth + strides - 1) // strides
            out_height = (height + strides - 1) // strides
            out_width = (width + strides - 1) // strides
        else:
            out_depth = (depth - kernel_size) // strides + 1
            out_height = (height - kernel_size) // strides + 1
            out_width = (width - kernel_size) // strides + 1

        return TensorShape(
            dims=[batch, out_depth, out_height, out_width, int(filters)],
            description="3D convolved output (NDHWC)"
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
            5,
            "[batch, depth, height, width, channels]"
        )
