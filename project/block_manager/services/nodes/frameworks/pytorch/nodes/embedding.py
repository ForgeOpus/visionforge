"""PyTorch Embedding Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchNodeDefinition


class EmbeddingNode(PyTorchNodeDefinition):
    """
    Embedding Layer.

    Converts integer indices to dense vectors.

    Input: [batch_size, seq_length] (integer indices)
    Output: [batch_size, seq_length, embedding_dim]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="embedding",
            label="Embedding",
            category="advanced",
            color="var(--color-purple)",
            icon="Table",
            description="Embedding layer for discrete tokens",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="num_embeddings",
                label="Vocabulary Size",
                type="number",
                required=True,
                min=1,
                description="Size of the vocabulary"
            ),
            ConfigField(
                name="embedding_dim",
                label="Embedding Dimension",
                type="number",
                required=True,
                min=1,
                description="Dimension of embedding vectors"
            ),
            ConfigField(
                name="padding_idx",
                label="Padding Index",
                type="number",
                description="Index for padding token (optional)"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 2:
            return None

        batch, seq_length = input_shape.dims
        embedding_dim = config.get("embedding_dim")
        if embedding_dim is None:
            return None

        return TensorShape(
            dims=[batch, seq_length, int(embedding_dim)],
            description="Embedded tokens"
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
            2,
            "[batch, seq_length]"
        )
