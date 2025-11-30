"""TensorFlow Input Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowSourceNode


class InputNode(TensorFlowSourceNode):
    """
    Input layer for the neural network.

    Defines the shape of input tensors flowing into the model.
    TensorFlow uses NHWC format (batch, height, width, channels).

    Output: [batch_size, height, width, channels] or [batch_size, features]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="input",
            label="Input",
            category="input",
            color="var(--color-teal)",
            icon="Download",
            description="Network input layer",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="shape",
                label="Input Shape",
                type="string",
                default="[1, 224, 224, 3]",
                description="Input tensor shape (NHWC format)"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Priority: DataLoader shape > manual config > default
        if input_shape:
            return input_shape

        # Parse shape from config
        shape_str = config.get("shape", "[1, 224, 224, 3]")
        dims = self.parse_shape_string(shape_str)

        if dims:
            return TensorShape(
                dims=dims,
                description="Input tensor (NHWC)"
            )

        # Fallback to default (NHWC format)
        return TensorShape(
            dims=[1, 224, 224, 3],
            description="Input tensor (NHWC)"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Input nodes only accept from DataLoader
        if source_node_type != "dataloader":
            return "Input nodes can only connect from DataLoader"
        return None
