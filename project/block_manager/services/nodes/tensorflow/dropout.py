"""TensorFlow Dropout Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class DropoutNode(NodeDefinition):
    """Dropout Layer using tf.keras.layers.Dropout"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="dropout",
            label="Dropout",
            category="basic",
            color="var(--color-gray)",
            icon="CircleSlash",
            description="Dropout regularization",
            framework=Framework.TENSORFLOW
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="rate",
                label="Dropout Rate",
                type="number",
                default=0.5,
                min=0.0,
                max=1.0,
                description="Fraction of inputs to drop (0.0 - 1.0)"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Dropout preserves input shape
        if not input_shape:
            return None
        
        return TensorShape(
            dims=input_shape.dims,
            description="Dropout applied"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Dropout accepts any input
        return None
    def get_tensorflow_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate TensorFlow code specification for Dropout layer"""
        rate = config.get('rate', 0.5)

        sanitized_id = node_id.replace('-', '_')
        class_name = 'DropoutLayer'
        layer_var = f'{sanitized_id}_DropoutLayer'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='dropout',
            node_id=node_id,
            init_params={'rate': rate},
            config_params=config,
            input_shape_info={},
            output_shape_info={},
            template_context={'rate': rate}
        )

