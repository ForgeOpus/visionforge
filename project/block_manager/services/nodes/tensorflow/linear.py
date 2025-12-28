"""TensorFlow Dense (Linear) Layer Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


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
    def get_tensorflow_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate TensorFlow code specification for Dense/Linear layer"""
        out_features = config.get('units', 128)
        use_bias = config.get('use_bias', True)
        in_features = input_shape.dims[1] if input_shape and len(input_shape.dims) >= 2 else 512

        sanitized_id = node_id.replace('-', '_')
        class_name = 'DenseLayer'
        layer_var = f'{sanitized_id}_DenseLayer'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='linear',
            node_id=node_id,
            init_params={'units': out_features, 'use_bias': use_bias},
            config_params=config,
            input_shape_info={'in_features': in_features},
            output_shape_info={'out_features': out_features},
            template_context={
                'in_features': in_features,
                'out_features': out_features,
                'use_bias': use_bias
            }
        )

