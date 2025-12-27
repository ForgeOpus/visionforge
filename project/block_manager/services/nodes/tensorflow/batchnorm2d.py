"""TensorFlow BatchNormalization Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class BatchNorm2DNode(NodeDefinition):
    """Batch Normalization using tf.keras.layers.BatchNormalization"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="batchnorm2d",
            label="BatchNorm2D",
            category="basic",
            color="var(--color-orange)",
            icon="Zap",
            description="Batch normalization layer",
            framework=Framework.TENSORFLOW
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="momentum",
                label="Momentum",
                type="number",
                default=0.99,
                min=0.0,
                max=1.0,
                description="Momentum for moving average"
            ),
            ConfigField(
                name="epsilon",
                label="Epsilon",
                type="number",
                default=0.001,
                min=0.0,
                description="Small constant for numerical stability"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # BatchNorm preserves input shape
        if not input_shape:
            return None
        
        return TensorShape(
            dims=input_shape.dims,
            description="Batch normalized"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Allow connections from input/dataloader
        if source_node_type in ("input", "dataloader", "empty", "custom"):
            return None
        
        # BatchNorm typically expects at least 2D input
        if source_output_shape and len(source_output_shape.dims) < 2:
            return "BatchNormalization requires input with at least 2 dimensions"
        
        return None
    def get_tensorflow_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate TensorFlow code specification for BatchNormalization layer"""
        epsilon = config.get('epsilon', 1e-5)
        momentum = config.get('momentum', 0.99)

        sanitized_id = node_id.replace('-', '_')
        class_name = 'BatchNormBlock'
        layer_var = f'{sanitized_id}_BatchNormBlock'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='batchnorm',
            node_id=node_id,
            init_params={'epsilon': epsilon, 'momentum': momentum},
            config_params=config,
            input_shape_info={},
            output_shape_info={},
            template_context={'epsilon': epsilon, 'momentum': momentum}
        )

