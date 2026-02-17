"""PyTorch AdaptiveAvgPool2D Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class AdaptiveAvgPool2DNode(NodeDefinition):
    """Adaptive Average Pooling layer that outputs specified size"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="adaptiveavgpool2d",
            label="AdaptiveAvgPool2D",
            category="basic",
            color="var(--color-primary)",
            icon="Resize",
            description="Adaptive average pooling to fixed output size",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="output_size",
                label="Output Size",
                type="string",
                default="1",
                description="Target output size (single number or [H, W])"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        if not input_shape or len(input_shape.dims) != 4:
            return None
        
        batch, channels = input_shape.dims[:2]
        output_size_str = str(config.get("output_size", "1"))
        
        # Parse output size
        try:
            if "," in output_size_str or "[" in output_size_str:
                # Parse as array [H, W]
                output_size_str = output_size_str.strip("[]()").replace(" ", "")
                parts = output_size_str.split(",")
                out_height = int(parts[0])
                out_width = int(parts[1]) if len(parts) > 1 else out_height
            else:
                # Single value means square output
                out_height = out_width = int(output_size_str)
        except (ValueError, IndexError):
            out_height = out_width = 1
        
        return TensorShape(
            dims=[batch, channels, out_height, out_width],
            description=f"AdaptiveAvgPool2D({out_height}x{out_width})"
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

        # Validate 4D input (N, C, H, W)
        return self.validate_dimensions(
            source_output_shape,
            4,
            "[batch, channels, height, width]"
        )

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """Generate PyTorch code specification for AdaptiveAvgPool2D layer"""
        output_size = config.get('output_size', '1')

        sanitized_id = node_id.replace('-', '_')
        class_name = 'AdaptiveAvgPool2DBlock'
        layer_var = f'{sanitized_id}_AdaptiveAvgPool2DBlock'

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var,
            node_type='adaptiveavgpool2d',
            node_id=node_id,
            init_params={
                'output_size': output_size
            },
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={
                'output_size': output_size
            }
        )
