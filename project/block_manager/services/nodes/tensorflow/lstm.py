"""TensorFlow LSTM Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class LSTMNode(NodeDefinition):
    """LSTM Layer using tf.keras.layers.LSTM"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="lstm",
            label="LSTM",
            category="recurrent",
            color="var(--color-green)",
            icon="Repeat",
            description="Long Short-Term Memory layer",
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
                default=False,
                description="Return full sequence or just last output"
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
                description="Dropout rate for recurrent connections"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or not config.get("units"):
            return None
        
        # LSTM expects: [batch, timesteps, features]
        if len(input_shape.dims) != 3:
            return None
        
        batch, timesteps, _ = input_shape.dims
        units = int(config["units"])
        return_sequences = config.get("return_sequences", False)
        
        if return_sequences:
            # Return full sequence: [batch, timesteps, units]
            return TensorShape(
                dims=[batch, timesteps, units],
                description="LSTM sequence output"
            )
        else:
            # Return only last output: [batch, units]
            return TensorShape(
                dims=[batch, units],
                description="LSTM final output"
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
            "[batch, timesteps, features]"
        )
