"""TensorFlow Loss Function Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import (
    ConfigField,
    ConfigOption,
    Framework,
    InputPort,
    NodeMetadata,
    TensorShape,
)
from ..base import TensorFlowTerminalNode


class LossNode(TensorFlowTerminalNode):
    """
    Loss function for training the neural network.

    Supports various loss types with appropriate input ports.
    """

    # Class attribute for backward compatibility
    input_ports_config = {
        "categorical_crossentropy": ["y_pred", "y_true"],
        "sparse_categorical_crossentropy": ["y_pred", "y_true"],
        "mse": ["y_pred", "y_true"],
        "mae": ["y_pred", "y_true"],
        "binary_crossentropy": ["y_pred", "y_true"],
        "kl_divergence": ["y_pred", "y_true"],
    }

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="loss",
            label="Loss Function",
            category="output",
            color="var(--color-red)",
            icon="Target",
            description="Loss function for training",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="loss_type",
                label="Loss Type",
                type="select",
                default="categorical_crossentropy",
                options=[
                    ConfigOption(value="categorical_crossentropy", label="Categorical Cross Entropy"),
                    ConfigOption(value="sparse_categorical_crossentropy", label="Sparse Categorical Cross Entropy"),
                    ConfigOption(value="mse", label="Mean Squared Error"),
                    ConfigOption(value="mae", label="Mean Absolute Error"),
                    ConfigOption(value="binary_crossentropy", label="Binary Cross Entropy"),
                    ConfigOption(value="kl_divergence", label="KL Divergence"),
                ],
                description="Type of loss function"
            )
        ]

    @property
    def input_ports(self) -> List[InputPort]:
        return [
            InputPort(
                id="y_pred",
                label="Predictions",
                description="Model predictions"
            ),
            InputPort(
                id="y_true",
                label="Ground Truth",
                description="True labels"
            ),
        ]

    def allows_multiple_inputs(self) -> bool:
        return True

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        return TensorShape(
            dims=[1],
            description="Loss value (scalar)"
        )
