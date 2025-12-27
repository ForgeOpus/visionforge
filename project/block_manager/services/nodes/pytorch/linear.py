"""PyTorch Linear Layer Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class LinearNode(NodeDefinition):
    """Linear/Fully Connected Layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="linear",
            label="Linear",
            category="basic",
            color="var(--color-primary)",
            icon="Lightning",
            description="Fully connected layer",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="out_features",
                label="Output Features",
                type="number",
                required=True,
                min=1,
                description="Number of output features"
            ),
            ConfigField(
                name="bias",
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
        if not input_shape or not config.get("out_features"):
            return None
        
        if len(input_shape.dims) != 2:
            return None
        
        return TensorShape(
            dims=[input_shape.dims[0], int(config["out_features"])],
            description="Fully connected output"
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
        
        # Validate dimension requirement
        return self.validate_dimensions(
            source_output_shape,
            2,
            "[batch, features]"
        )

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate PyTorch code specification for Linear layer"""
        out_features = config.get('out_features', 128)
        bias = config.get('bias', True)

        in_features = input_shape.dims[1] if input_shape and len(input_shape.dims) >= 2 else 512

        sanitized_id = node_id.replace('-', '_')
        class_name = 'LinearLayer'
        layer_var = f'{sanitized_id}_LinearLayer'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='linear',
            node_id=node_id,
            init_params={
                'in_features': in_features,
                'out_features': out_features,
                'bias': bias
            },
            config_params=config,
            input_shape_info={
                'in_features': in_features,
                'dims': input_shape.dims if input_shape else []
            },
            output_shape_info={
                'out_features': out_features,
                'dims': output_shape.dims if output_shape else []
            },
            template_context={
                'in_features': in_features,
                'out_features': out_features,
                'bias': bias
            }
        )
