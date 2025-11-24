"""PyTorch Multi-Head Attention Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchNodeDefinition


class AttentionNode(PyTorchNodeDefinition):
    """
    Multi-head self-attention mechanism.

    Input: [batch_size, seq_length, embed_dim]
    Output: [batch_size, seq_length, embed_dim]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="attention",
            label="Multi-Head Attention",
            category="advanced",
            color="var(--color-purple)",
            icon="Brain",
            description="Multi-head self-attention mechanism",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="embed_dim",
                label="Embedding Dimension",
                type="number",
                required=True,
                min=1,
                description="Total dimension of the model"
            ),
            ConfigField(
                name="num_heads",
                label="Number of Heads",
                type="number",
                default=8,
                min=1,
                description="Number of attention heads"
            ),
            ConfigField(
                name="dropout",
                label="Dropout",
                type="number",
                default=0.0,
                min=0.0,
                max=1.0,
                description="Dropout probability"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 3:
            return None

        # Output shape is same as input for self-attention
        return TensorShape(
            dims=input_shape.dims.copy(),
            description="Attention output"
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
            "[batch, seq_length, embed_dim]"
        )
