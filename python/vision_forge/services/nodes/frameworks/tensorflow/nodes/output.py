"""TensorFlow Output Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, ConfigOption, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowTerminalNode


class OutputNode(TensorFlowTerminalNode):
    """
    Output Layer / Loss Function.

    Defines the model output and loss.

    Input: any shape
    Output: None (terminal node)
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="output",
            label="Output",
            category="output",
            color="var(--color-red)",
            icon="Upload",
            description="Model output / loss function",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="loss",
                label="Loss Function",
                type="select",
                default="categorical_crossentropy",
                options=[
                    ConfigOption(value="categorical_crossentropy", label="Categorical Crossentropy"),
                    ConfigOption(value="sparse_categorical_crossentropy", label="Sparse Categorical Crossentropy"),
                    ConfigOption(value="binary_crossentropy", label="Binary Crossentropy"),
                    ConfigOption(value="mse", label="Mean Squared Error"),
                    ConfigOption(value="mae", label="Mean Absolute Error")
                ],
                description="Loss function for training"
            ),
            ConfigField(
                name="num_classes",
                label="Number of Classes",
                type="number",
                default=10,
                min=1,
                description="Number of output classes"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Terminal node - just pass through input shape
        return input_shape

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Accept any valid input
        if source_node_type in ("empty", "custom"):
            return None

        if not source_output_shape:
            return None

        return None
