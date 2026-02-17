"""PyTorch LSTM Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class LSTMNode(NodeDefinition):
    """Long Short-Term Memory recurrent layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="lstm",
            label="LSTM",
            category="advanced",
            color="var(--color-purple)",
            icon="ArrowsClockwise",
            description="LSTM recurrent layer",
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
                description="Use bidirectional LSTM"
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
            description=f"LSTM({hidden_size})"
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

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate PyTorch code specification for LSTM layer"""
        hidden_size = config.get('hidden_size', 128)
        num_layers = config.get('num_layers', 1)
        bias = config.get('bias', True)
        batch_first = config.get('batch_first', True)
        dropout = config.get('dropout', 0.0)
        bidirectional = config.get('bidirectional', False)

        # Determine input_size from input shape if available
        input_size = None
        if input_shape and len(input_shape.dims) == 3:
            if batch_first:
                input_size = input_shape.dims[2]
            else:
                input_size = input_shape.dims[2]

        sanitized_id = node_id.replace('-', '_')
        class_name = 'LSTMBlock'
        layer_var = f'{sanitized_id}_LSTMBlock'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='lstm',
            node_id=node_id,
            init_params={
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'bias': bias,
                'batch_first': batch_first,
                'dropout': dropout,
                'bidirectional': bidirectional
            },
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'bias': bias,
                'batch_first': batch_first,
                'dropout': dropout,
                'bidirectional': bidirectional
            }
        )
