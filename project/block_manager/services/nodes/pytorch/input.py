"""PyTorch Input Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


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
