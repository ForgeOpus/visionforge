"""PyTorch Multi-Head Attention Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class AttentionNode(NodeDefinition):
    """Multi-Head Attention layer"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="attention",
            label="Multi-Head Attention",
            category="advanced",
            color="var(--color-accent)",
            icon="Zap",
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
                default=512,
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
            ),
            ConfigField(
                name="bias",
                label="Use Bias",
                type="boolean",
                default=True,
                description="Whether to use bias in linear projections"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Multi-head attention preserves shape
        if input_shape:
            return TensorShape(
                dims=input_shape.dims,
                description="Attention output"
            )
        return None

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Attention accepts any input shape (typically 3D: [batch, seq_len, embed_dim])
        return None

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate PyTorch code specification for Multi-Head Attention layer"""
        embed_dim = config.get('embed_dim', 512)
        num_heads = config.get('num_heads', 8)
        dropout = config.get('dropout', 0.0)
        bias = config.get('bias', True)

        sanitized_id = node_id.replace('-', '_')
        class_name = 'MultiHeadAttentionLayer'
        layer_var = f'{sanitized_id}_MultiHeadAttentionLayer'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='attention',
            node_id=node_id,
            init_params={
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'dropout': dropout,
                'bias': bias
            },
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'dropout': dropout,
                'bias': bias
            }
        )
