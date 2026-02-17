"""PyTorch Input Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class InputNode(NodeDefinition):
    """Input layer for the neural network"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="input",
            label="Input",
            category="input",
            color="var(--color-teal)",
            icon="Download",
            description="Network input layer",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="shape",
                label="Input Shape",
                type="string",
                default="[1, 3, 224, 224]",
                description="Input tensor shape as JSON array (e.g., [1, 3, 224, 224]). Overridden by DataLoader if connected."
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Priority: DataLoader shape > manual config > default
        if input_shape:
            return input_shape
        
        # Parse shape from config
        shape_str = config.get("shape", "[1, 3, 224, 224]")
        dims = self.parse_shape_string(shape_str)
        
        if dims:
            return TensorShape(
                dims=dims,
                description="Input tensor"
            )
        
        # Fallback to default
        return TensorShape(
            dims=[1, 3, 224, 224],
            description="Input tensor"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Input nodes typically don't accept incoming connections
        # except from DataLoader
        if source_node_type != "dataloader":
            return "Input nodes can only connect from DataLoader"
        return None

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Source node - doesn't generate layer code. For interface compatibility."""
        sanitized_id = node_id.replace('-', '_')
        return LayerCodeSpec(
            class_name='SourceNode',
            layer_variable_name=f'{sanitized_id}_Source',
            node_type='input',
            node_id=node_id,
            init_params={},
            config_params=config,
            input_shape_info={'dims': []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={}
        )
