"""PyTorch LSTM Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchNodeDefinition


class LSTMNode(PyTorchNodeDefinition):
    """
    Long Short-Term Memory (LSTM) Layer.

    Input: [batch_size, seq_length, input_size]
    Output: [batch_size, seq_length, hidden_size * num_directions]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="lstm",
            label="LSTM",
            category="advanced",
            color="var(--color-purple)",
            icon="Repeat",
            description="Long Short-Term Memory layer",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="hidden_size",
                label="Hidden Size",
                type="number",
                required=True,
                min=1,
                description="Number of features in hidden state"
            ),
            ConfigField(
                name="num_layers",
                label="Number of Layers",
                type="number",
                default=1,
                min=1,
                description="Number of recurrent layers"
            ),
            ConfigField(
                name="bidirectional",
                label="Bidirectional",
                type="boolean",
                default=False,
                description="Use bidirectional LSTM"
            ),
            ConfigField(
                name="dropout",
                label="Dropout",
                type="number",
                default=0.0,
                min=0.0,
                max=1.0,
                description="Dropout probability between layers"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 3:
            return None

        batch, seq_length, _ = input_shape.dims
        hidden_size = config.get("hidden_size")
        if hidden_size is None:
            return None

        bidirectional = config.get("bidirectional", False)
        num_directions = 2 if bidirectional else 1

        return TensorShape(
            dims=[batch, seq_length, int(hidden_size) * num_directions],
            description="LSTM output"
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
            "[batch, seq_length, features]"
        )
