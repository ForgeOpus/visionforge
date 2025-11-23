"""PyTorch Conv3D Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchNodeDefinition


class Conv3DNode(PyTorchNodeDefinition):
    """
    3D Convolutional Layer.

    Input: [batch_size, in_channels, depth, height, width]
    Output: [batch_size, out_channels, out_depth, out_height, out_width]
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
                description="Zero-padding added to all sides"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 5:
            return None

        batch, _, depth, height, width = input_shape.dims
        out_channels = config.get("out_channels")
        if out_channels is None:
            return None

        kernel_size = int(config.get("kernel_size", 3))
        stride = int(config.get("stride", 1))
        padding = int(config.get("padding", 0))

        # Compute output dimensions (same formula for all spatial dims)
        out_depth = (depth + 2 * padding - kernel_size) // stride + 1
        out_height = (height + 2 * padding - kernel_size) // stride + 1
        out_width = (width + 2 * padding - kernel_size) // stride + 1

        return TensorShape(
            dims=[batch, int(out_channels), out_depth, out_height, out_width],
            description="3D convolved output"
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
            "[batch, channels, depth, height, width]"
        )
