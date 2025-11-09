"""PyTorch Add Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class AddNode(NodeDefinition):
    """Element-wise addition of multiple tensors"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="add",
            label="Add",
            category="merge",
            color="var(--color-accent)",
            icon="Plus",
            description="Element-wise addition",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return []  # Add operation doesn't need configuration
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Element-wise addition preserves shape
        # All inputs must have the same shape
        if not input_shape:
            return None
        
        return TensorShape(
            dims=input_shape.dims,
            description="Element-wise sum"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Add accepts multiple inputs - validation happens at graph level
        # to ensure all inputs have the same shape
        return None
    
    @property
    def allows_multiple_inputs(self) -> bool:
        """Add nodes accept multiple input connections"""
        return True
