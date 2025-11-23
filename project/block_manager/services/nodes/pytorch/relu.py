"""PyTorch ReLU Activation Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class ReLUNode(NodeDefinition):
    """ReLU (Rectified Linear Unit) Activation Function"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="relu",
            label="ReLU",
            category="activation",
            color="#f59e0b",
            icon="Lightning",
            description="Rectified Linear Unit activation function",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="inplace",
                label="Inplace Operation",
                type="boolean",
                default=False,
                description="Perform operation in-place to save memory"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # ReLU preserves input shape
        if not input_shape:
            return None
        
        return TensorShape(
            dims=input_shape.dims.copy(),
            description="ReLU activation output"
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
        
        # ReLU accepts any numeric tensor
        if not source_output_shape:
            return "ReLU requires a valid input shape"
        
        return None
