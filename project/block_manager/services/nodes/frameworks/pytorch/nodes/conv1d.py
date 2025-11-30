"""PyTorch Conv1D Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchNodeDefinition


class Conv1DNode(PyTorchNodeDefinition):
    """
    1D Convolutional Layer.

    Input: [batch_size, in_channels, length]
    Output: [batch_size, out_channels, output_length]
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
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="out_channels",
                label="Output Channels",
                type="number",
                required=True,
                min=1,
                description="Number of output channels"
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
                name="stride",
                label="Stride",
                type="number",
                default=1,
                min=1,
                description="Stride of convolution"
            ),
            ConfigField(
                name="padding",
                label="Padding",
                type="number",
                default=0,
                min=0,
                description="Zero-padding added to both sides"
            ),
            ConfigField(
                name="dilation",
                label="Dilation",
                type="number",
                default=1,
                min=1,
                description="Spacing between kernel elements"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 3:
            return None

        batch, _, length = input_shape.dims
        out_channels = config.get("out_channels")
        if out_channels is None:
            return None

        kernel_size = int(config.get("kernel_size", 3))
        stride = int(config.get("stride", 1))
        padding = int(config.get("padding", 0))
        dilation = int(config.get("dilation", 1))

        out_length = self.compute_conv1d_output(
            length, kernel_size, stride, padding, dilation
        )

        return TensorShape(
            dims=[batch, int(out_channels), out_length],
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
            "[batch, channels, length]"
        )
