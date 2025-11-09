"""TensorFlow Embedding Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class EmbeddingNode(NodeDefinition):
    """Embedding Layer using tf.keras.layers.Embedding"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="embedding",
            label="Embedding",
            category="basic",
            color="var(--color-purple)",
            icon="Hash",
            description="Embedding layer for categorical inputs",
            framework=Framework.TENSORFLOW
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="input_dim",
                label="Input Dim (Vocabulary Size)",
                type="number",
                required=True,
                min=1,
                description="Size of vocabulary (number of unique tokens)"
            ),
            ConfigField(
                name="output_dim",
                label="Output Dim (Embedding Size)",
                type="number",
                required=True,
                min=1,
                description="Dimension of embedding vectors"
            ),
            ConfigField(
                name="mask_zero",
                label="Mask Zero",
                type="boolean",
                default=False,
                description="Whether to mask zero values (for padding)"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or not config.get("output_dim"):
            return None
        
        # Embedding expects: [batch, sequence_length]
        # Output: [batch, sequence_length, embedding_dim]
        if len(input_shape.dims) < 2:
            return None
        
        output_dims = input_shape.dims + [int(config["output_dim"])]
        
        return TensorShape(
            dims=output_dims,
            description="Embedded sequences"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        if source_node_type in ("input", "dataloader", "empty", "custom"):
            return None
        
        # Embedding typically expects integer input with at least 2 dimensions
        if source_output_shape and len(source_output_shape.dims) < 2:
            return "Embedding requires input with at least 2 dimensions [batch, sequence_length]"
        
        return None
