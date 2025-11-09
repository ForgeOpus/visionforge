"""PyTorch GRU Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class GRUNode(NodeDefinition):
    """Gated Recurrent Unit layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="gru",
            label="GRU",
            category="advanced",
            color="var(--color-purple)",
            icon="ArrowsClockwise",
            description="GRU recurrent layer",
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
                label="Layers",
                type="number",
                default=1,
                min=1,
                description="Number of recurrent layers"
            ),
            ConfigField(
                name="bias",
                label="Use Bias",
                type="boolean",
                default=True,
                description="Use bias weights"
            ),
            ConfigField(
                name="batch_first",
                label="Batch First",
                type="boolean",
                default=True,
                description="Input shape is (batch, seq, feature)"
            ),
            ConfigField(
                name="dropout",
                label="Dropout",
                type="number",
                default=0.0,
                min=0.0,
                max=1.0,
                description="Dropout probability (if layers > 1)"
            ),
            ConfigField(
                name="bidirectional",
                label="Bidirectional",
                type="boolean",
                default=False,
                description="Use bidirectional GRU"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 3:
            return None
        
        batch_first = config.get("batch_first", True)
        hidden_size = int(config.get("hidden_size", 128))
        bidirectional = config.get("bidirectional", False)
        
        if batch_first:
            batch, seq_len, _ = input_shape.dims
        else:
            seq_len, batch, _ = input_shape.dims
        
        # Output size is doubled if bidirectional
        out_features = hidden_size * (2 if bidirectional else 1)
        
        if batch_first:
            dims = [batch, seq_len, out_features]
        else:
            dims = [seq_len, batch, out_features]
        
        return TensorShape(
            dims=dims,
            description=f"GRU({hidden_size})"
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
        
        # Validate 3D input (batch, seq, features) or (seq, batch, features)
        return self.validate_dimensions(
            source_output_shape,
            3,
            "[batch, sequence, features] or [sequence, batch, features]"
        )
