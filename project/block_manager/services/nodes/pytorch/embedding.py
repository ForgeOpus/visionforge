"""PyTorch Embedding Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class EmbeddingNode(NodeDefinition):
    """Embedding layer for discrete tokens"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="embedding",
            label="Embedding",
            category="advanced",
            color="var(--color-purple)",
            icon="TextAa",
            description="Token embedding layer",
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
                description="Size of each embedding vector"
            ),
            ConfigField(
                name="padding_idx",
                label="Padding Index",
                type="number",
                default=-1,
                description="Padding token index (or -1 for none)"
            ),
            ConfigField(
                name="max_norm",
                label="Max Norm",
                type="number",
                default=0,
                min=0,
                description="Renormalize embeddings (0 for no normalization)"
            ),
            ConfigField(
                name="scale_grad_by_freq",
                label="Scale Grad by Freq",
                type="boolean",
                default=False,
                description="Scale gradients by word frequency"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape:
            return None
        
        embedding_dim = int(config.get("embedding_dim", 128))
        
        # Input is typically (batch, sequence_length) of token indices
        # Output is (batch, sequence_length, embedding_dim)
        if len(input_shape.dims) == 2:
            batch, seq_len = input_shape.dims
            return TensorShape(
                dims=[batch, seq_len, embedding_dim],
                description=f"Embedding({embedding_dim})"
            )
        
        # If input has more dimensions, append embedding_dim
        return TensorShape(
            dims=input_shape.dims + [embedding_dim],
            description=f"Embedding({embedding_dim})"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Allow connections from input/dataloader without shape validation
        if source_node_type in ("input", "dataloader"):
            return None

        # Empty and custom nodes are flexible
        if source_node_type in ("empty", "custom"):
            return None

        # Embedding typically expects integer indices, shape validation is lenient
        return None

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate PyTorch code specification for Embedding layer"""
        num_embeddings = config.get('num_embeddings', 1000)
        embedding_dim = config.get('embedding_dim', 128)
        padding_idx = config.get('padding_idx', -1)
        max_norm = config.get('max_norm', 0)
        scale_grad_by_freq = config.get('scale_grad_by_freq', False)

        sanitized_id = node_id.replace('-', '_')
        class_name = 'EmbeddingBlock'
        layer_var = f'{sanitized_id}_EmbeddingBlock'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='embedding',
            node_id=node_id,
            init_params={
                'num_embeddings': num_embeddings,
                'embedding_dim': embedding_dim,
                'padding_idx': padding_idx if padding_idx >= 0 else None,
                'max_norm': max_norm if max_norm > 0 else None,
                'scale_grad_by_freq': scale_grad_by_freq
            },
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={
                'num_embeddings': num_embeddings,
                'embedding_dim': embedding_dim,
                'padding_idx': padding_idx,
                'max_norm': max_norm,
                'scale_grad_by_freq': scale_grad_by_freq
            }
        )
