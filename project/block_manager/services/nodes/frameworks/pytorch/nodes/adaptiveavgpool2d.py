"""PyTorch AdaptiveAvgPool2D Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchNodeDefinition


class AdaptiveAvgPool2DNode(PyTorchNodeDefinition):
    """
    2D Adaptive Average Pooling Layer.

    Automatically adjusts pooling to produce desired output size.

    Input: [batch_size, channels, height, width]
    Output: [batch_size, channels, output_height, output_width]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="adaptiveavgpool2d",
            label="AdaptiveAvgPool2D",
            category="basic",
            color="var(--color-purple)",
            icon="SquaresFour",
            description="2D adaptive average pooling layer",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="output_size",
                label="Output Size",
                type="number",
                default=1,
                min=1,
                description="Target output size (height and width)"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 4:
            return None

        batch, channels, _, _ = input_shape.dims
        output_size = int(config.get("output_size", 1))

        return TensorShape(
            dims=[batch, channels, output_size, output_size],
            description="Adaptive average pooled output"
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
