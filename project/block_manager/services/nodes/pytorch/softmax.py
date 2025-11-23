"""PyTorch Softmax Activation Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class SoftmaxNode(NodeDefinition):
    """Softmax Activation Function"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="softmax",
            label="Softmax",
            category="activation",
            color="#f59e0b",
            icon="Function",
            description="Softmax activation function for probability distributions",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="dim",
                label="Dimension",
                type="number",
                default=-1,
                description="Dimension along which Softmax will be computed"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Softmax preserves input shape
        if not input_shape:
            return None
        
        return TensorShape(
            dims=input_shape.dims.copy(),
            description="Softmax activation output"
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
        
        # Softmax accepts any numeric tensor
        if not source_output_shape:
            return "Softmax requires a valid input shape"
        
        return None
