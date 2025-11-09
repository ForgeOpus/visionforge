"""PyTorch Dropout Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class DropoutNode(NodeDefinition):
    """Dropout regularization layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="dropout",
            label="Dropout",
            category="basic",
            color="var(--color-primary)",
            icon="Percent",
            description="Dropout regularization",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="p",
                label="Dropout Rate",
                type="number",
                default=0.5,
                min=0.0,
                max=1.0,
                description="Probability of dropping a unit (0 to 1)"
            ),
            ConfigField(
                name="inplace",
                label="In-place",
                type="boolean",
                default=False,
                description="Perform operation in-place"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Dropout preserves shape
        if input_shape:
            return TensorShape(
                dims=input_shape.dims,
                description=f"Dropout ({config.get('p', 0.5)})"
            )
        return None
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Dropout accepts any input shape
        return None
