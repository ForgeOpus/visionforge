"""PyTorch AvgPool2D Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchNodeDefinition


class AvgPool2DNode(PyTorchNodeDefinition):
    """
    2D Average Pooling Layer.

    Input: [batch_size, channels, height, width]
    Output: [batch_size, channels, pooled_height, pooled_width]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="avgpool2d",
            label="AvgPool2D",
            category="basic",
            color="var(--color-purple)",
            icon="SquaresFour",
            description="2D average pooling layer",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="kernel_size",
                label="Kernel Size",
                type="number",
                default=2,
                min=1,
                description="Size of pooling window"
            ),
            ConfigField(
                name="stride",
                label="Stride",
                type="number",
                default=2,
                min=1,
                description="Stride of pooling window"
            ),
            ConfigField(
                name="padding",
                label="Padding",
                type="number",
                default=0,
                min=0,
                description="Zero-padding added to both sides"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 4:
            return None

        batch, channels, height, width = input_shape.dims
        kernel_size = int(config.get("kernel_size", 2))
        stride = int(config.get("stride", 2))
        padding = int(config.get("padding", 0))

        out_height, out_width = self.compute_pool2d_output(
            height, width, kernel_size, stride, padding
        )

        return TensorShape(
            dims=[batch, channels, out_height, out_width],
            description="Average pooled feature map"
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
            "[batch, channels, height, width]"
        )
