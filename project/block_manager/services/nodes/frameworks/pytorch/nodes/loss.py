"""PyTorch Loss Function Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import (
    ConfigField,
    ConfigOption,
    Framework,
    InputPort,
    NodeMetadata,
    TensorShape,
)
from ..base import PyTorchTerminalNode


class LossNode(PyTorchTerminalNode):
    """
    Loss function for training the neural network.

    Supports various loss types with appropriate input ports
    for each loss function's requirements.
    """

    # Class attribute for backward compatibility with validation.py
    input_ports_config = {
        "cross_entropy": ["y_pred", "y_true"],
        "mse": ["y_pred", "y_true"],
        "mae": ["y_pred", "y_true"],
        "bce": ["y_pred", "y_true"],
        "nll": ["y_pred", "y_true"],
        "kl_div": ["y_pred", "y_true"],
        "triplet": ["anchor", "positive", "negative"],
        "contrastive": ["input1", "input2", "label"],
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
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="loss_type",
                label="Loss Type",
                type="select",
                default="cross_entropy",
                options=[
                    ConfigOption(value="cross_entropy", label="Cross Entropy"),
                    ConfigOption(value="mse", label="Mean Squared Error"),
                    ConfigOption(value="mae", label="Mean Absolute Error"),
                    ConfigOption(value="bce", label="Binary Cross Entropy"),
                    ConfigOption(value="triplet", label="Triplet Loss"),
                    ConfigOption(value="contrastive", label="Contrastive Loss"),
                    ConfigOption(value="nll", label="Negative Log Likelihood"),
                    ConfigOption(value="kl_div", label="KL Divergence"),
                ],
                description="Type of loss function"
            )
        ]

    @property
    def input_ports(self) -> List[InputPort]:
        """Define the input ports for the loss function"""
        return [
            InputPort(
                id="y_pred",
                label="Predictions",
                description="Model predictions (y_pred)"
            ),
            InputPort(
                id="y_true",
                label="Ground Truth",
                description="True labels (y_true)"
            ),
        ]

    def allows_multiple_inputs(self) -> bool:
        """Loss nodes accept multiple inputs"""
        return True

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Loss functions produce a scalar output
        return TensorShape(
            dims=[1],
            description="Loss value (scalar)"
        )
