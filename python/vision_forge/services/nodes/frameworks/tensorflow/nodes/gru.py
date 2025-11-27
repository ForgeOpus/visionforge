"""TensorFlow GRU Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowNodeDefinition


class GRUNode(TensorFlowNodeDefinition):
    """
    Gated Recurrent Unit (GRU) Layer (Keras).

    Input: [batch_size, seq_length, features]
    Output: [batch_size, seq_length, units] or [batch_size, units]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="gru",
            label="GRU",
            category="advanced",
            color="var(--color-purple)",
            icon="Repeat",
            description="Gated Recurrent Unit layer",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="units",
                label="Units",
                type="number",
                required=True,
                min=1,
                description="Dimensionality of output space"
            ),
            ConfigField(
                name="return_sequences",
                label="Return Sequences",
                type="boolean",
                default=True,
                description="Return full sequence or last output"
            ),
            ConfigField(
                name="dropout",
                label="Dropout",
                type="number",
                default=0.0,
                min=0.0,
                max=1.0,
                description="Dropout rate for inputs"
            ),
            ConfigField(
                name="recurrent_dropout",
                label="Recurrent Dropout",
                type="number",
                default=0.0,
                min=0.0,
                max=1.0,
                description="Dropout rate for recurrent state"
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
        units = config.get("units")
        if units is None:
            return None

        return_sequences = config.get("return_sequences", True)

        if return_sequences:
            return TensorShape(
                dims=[batch, seq_length, int(units)],
                description="GRU sequence output"
            )
        else:
            return TensorShape(
                dims=[batch, int(units)],
                description="GRU final output"
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
