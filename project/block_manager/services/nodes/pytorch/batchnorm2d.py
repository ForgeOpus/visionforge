"""PyTorch BatchNorm2D Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class BatchNorm2DNode(NodeDefinition):
    """2D Batch Normalization layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="batchnorm2d",
            label="BatchNorm2D",
            category="basic",
            color="var(--color-primary)",
            icon="ChartLineUp",
            description="Batch normalization for 2D inputs",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="num_features",
                label="Number of Features",
                type="number",
                required=True,
                min=1,
                description="Number of channels (C from [N, C, H, W])"
            ),
            ConfigField(
                name="eps",
                label="Epsilon",
                type="number",
                default=1e-5,
                description="Value for numerical stability"
            ),
            ConfigField(
                name="momentum",
                label="Momentum",
                type="number",
                default=0.1,
                min=0.0,
                max=1.0,
                description="Momentum for running mean/variance"
            ),
            ConfigField(
                name="affine",
                label="Affine",
                type="boolean",
                default=True,
                description="Use learnable affine parameters"
            ),
            ConfigField(
                name="track_running_stats",
                label="Track Stats",
                type="boolean",
                default=True,
                description="Track running mean and variance"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # BatchNorm2D preserves shape
        if input_shape:
            return TensorShape(
                dims=input_shape.dims,
                description="Batch normalized"
            )
        return None
    
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
        
        # Validate 4D input (N, C, H, W)
        return self.validate_dimensions(
            source_output_shape,
            4,
            "[batch, channels, height, width]"
        )

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate PyTorch code specification for BatchNorm2D layer"""
        num_features = config.get('num_features') or (input_shape.dims[1] if input_shape and len(input_shape.dims) >= 2 else 64)
        eps = config.get('eps', 1e-5)
        momentum = config.get('momentum', 0.1)
        affine = config.get('affine', True)

        sanitized_id = node_id.replace('-', '_')
        class_name = 'BatchNormBlock'
        layer_var = f'{sanitized_id}_BatchNormBlock'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='batchnorm',
            node_id=node_id,
            init_params={
                'num_features': num_features,
                'eps': eps,
                'momentum': momentum,
                'affine': affine
            },
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={
                'num_features': num_features,
                'eps': eps,
                'momentum': momentum,
                'affine': affine
            }
        )
