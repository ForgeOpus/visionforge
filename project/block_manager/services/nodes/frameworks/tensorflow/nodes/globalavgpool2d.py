"""TensorFlow GlobalAveragePooling2D Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowNodeDefinition


class GlobalAvgPool2DNode(TensorFlowNodeDefinition):
    """
    Global Average Pooling 2D Layer (Keras).

    Equivalent to PyTorch's AdaptiveAvgPool2D with output_size=1.

    Input: [batch_size, height, width, channels] (NHWC)
    Output: [batch_size, channels]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="globalavgpool2d",
            label="GlobalAveragePooling2D",
            category="basic",
            color="var(--color-purple)",
            icon="SquaresFour",
            description="Global 2D average pooling layer",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return []  # No configuration needed

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 4:
            return None

        batch, _, _, channels = input_shape.dims

        return TensorShape(
            dims=[batch, channels],
            description="Global average pooled output"
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
