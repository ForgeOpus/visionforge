"""PyTorch Flatten Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class FlattenNode(NodeDefinition):
    """Flatten layer to convert multi-dimensional tensor to 2D"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="flatten",
            label="Flatten",
            category="basic",
            color="var(--color-primary)",
            icon="Rows",
            description="Flatten tensor to 2D",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="start_dim",
                label="Start Dimension",
                type="number",
                default=1,
                min=0,
                description="First dimension to flatten (default: 1, preserving batch)"
            ),
            ConfigField(
                name="end_dim",
                label="End Dimension",
                type="number",
                default=-1,
                description="Last dimension to flatten (-1 for all remaining)"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) < 2:
            return None
        
        start_dim = int(config.get("start_dim", 1))
        batch_size = input_shape.dims[0]
        
        # Calculate flattened features (multiply all dimensions after start_dim)
        features = 1
        for dim in input_shape.dims[start_dim:]:
            features *= dim
        
        return TensorShape(
            dims=[batch_size, features],
            description="Flattened tensor"
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
        
        # Validate that input has at least 2 dimensions
        if source_output_shape and len(source_output_shape.dims) < 2:
            return "Flatten requires input with at least 2 dimensions"
        
        return None
