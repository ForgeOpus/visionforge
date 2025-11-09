"""TensorFlow Dense (Linear) Layer Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class LinearNode(NodeDefinition):
    """Dense/Fully Connected Layer using tf.keras.layers.Dense"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="linear",
            label="Dense",
            category="basic",
            color="var(--color-primary)",
            icon="Lightning",
            description="Fully connected layer (Dense)",
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
                description="Number of output units (neurons)"
            ),
            ConfigField(
                name="activation",
                label="Activation",
                type="select",
                default="None",
                options=[
                    {"value": "None", "label": "None"},
                    {"value": "relu", "label": "ReLU"},
                    {"value": "sigmoid", "label": "Sigmoid"},
                    {"value": "tanh", "label": "Tanh"},
                    {"value": "softmax", "label": "Softmax"}
                ],
                description="Activation function"
            ),
            ConfigField(
                name="use_bias",
                label="Use Bias",
                type="boolean",
                default=True,
                description="Add learnable bias"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or not config.get("units"):
            return None
        
        if len(input_shape.dims) < 2:
            return None
        
        # Dense layer outputs [batch, units]
        # If input is higher dimensional, only last dimension changes
        output_dims = input_shape.dims[:-1] + [int(config["units"])]
        
        return TensorShape(
            dims=output_dims,
            description="Dense layer output"
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
        
        # Validate dimension requirement (at least 2D)
        if source_output_shape and len(source_output_shape.dims) < 2:
            return "Dense layer requires input with at least 2 dimensions [batch, features, ...]"
        
        return None
