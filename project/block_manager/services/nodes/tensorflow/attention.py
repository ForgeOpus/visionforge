"""TensorFlow Multi-Head Attention Node Definition"""

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
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="num_heads",
                label="Number of Heads",
                type="number",
                default=8,
                min=1,
                description="Number of attention heads"
            ),
            ConfigField(
                name="key_dim",
                label="Key Dimension",
                type="number",
                default=64,
                min=1,
                description="Size of each attention head for query and key"
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
        # Attention accepts any input shape
        return None

    def get_tensorflow_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate TensorFlow code specification for Multi-Head Attention layer"""
        num_heads = config.get('num_heads', 8)
        key_dim = config.get('key_dim', 64)
        dropout = config.get('dropout', 0.0)

        sanitized_id = node_id.replace('-', '_')
        class_name = 'MultiHeadAttentionLayer'
        layer_var = f'{sanitized_id}_MultiHeadAttentionLayer'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='attention',
            node_id=node_id,
            init_params={
                'num_heads': num_heads,
                'key_dim': key_dim,
                'dropout': dropout
            },
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={
                'num_heads': num_heads,
                'key_dim': key_dim,
                'dropout': dropout
            }
        )
