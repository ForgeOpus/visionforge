"""PyTorch Custom Layer Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class CustomNode(NodeDefinition):
    """Custom User-Defined Layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="custom",
            label="Custom Layer",
            category="custom",
            color="#6b7280",
            icon="Code",
            description="User-defined custom layer with arbitrary PyTorch code",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="class_name",
                label="Class Name",
                type="text",
                required=True,
                description="Name of the custom layer class"
            ),
            ConfigField(
                name="code",
                label="Layer Code",
                type="textarea",
                required=True,
                description="PyTorch layer implementation code"
            ),
            ConfigField(
                name="output_shape",
                label="Output Shape",
                type="text",
                required=True,
                description="Expected output shape as JSON array (e.g., [32, 64, 64])"
            ),
            ConfigField(
                name="init_args",
                label="Init Arguments",
                type="textarea",
                default="{}",
                description="Constructor arguments as JSON object"
            ),
            ConfigField(
                name="description",
                label="Description",
                type="textarea",
                default="",
                description="Optional description of the custom layer"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Use user-provided output shape
        output_shape_str = config.get("output_shape")
        if not output_shape_str:
            return None
        
        try:
            # Parse output shape from JSON string
            import json
            if isinstance(output_shape_str, str):
                dims = json.loads(output_shape_str)
            else:
                dims = output_shape_str
            
            if not isinstance(dims, list):
                return None
            
            return TensorShape(
                dims=dims,
                description=f"Custom layer output: {config.get('class_name', 'CustomLayer')}"
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Custom layers are permissive - accept any connection
        # User is responsible for ensuring compatibility
        
        # Basic validation: ensure source has some output
        if source_node_type not in ("input", "dataloader", "empty", "custom"):
            if not source_output_shape:
                return "Custom layer requires a valid input shape from source"
        
        # Warn if output_shape is not configured
        if not target_config.get("output_shape"):
            return "Custom layer requires output_shape to be configured"
        
        return None
