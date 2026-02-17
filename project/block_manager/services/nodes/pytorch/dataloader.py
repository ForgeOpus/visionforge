"""PyTorch DataLoader Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class DataLoaderNode(NodeDefinition):
    """DataLoader for feeding data to the network"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="dataloader",
            label="DataLoader",
            category="input",
            color="var(--color-teal)",
            icon="Database",
            description="Data loading and batching",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="batch_size",
                label="Batch Size",
                type="number",
                default=32,
                min=1,
                description="Number of samples per batch"
            ),
            ConfigField(
                name="shuffle",
                label="Shuffle",
                type="boolean",
                default=True,
                description="Shuffle data each epoch"
            ),
            ConfigField(
                name="num_workers",
                label="Workers",
                type="number",
                default=0,
                min=0,
                description="Number of data loading workers"
            ),
            ConfigField(
                name="output_shape",
                label="Output Shape",
                type="string",
                default="[32, 3, 224, 224]",
                description="Shape of batched data (e.g., [batch, channels, height, width])"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Parse output shape from config
        shape_str = config.get("output_shape", "[32, 3, 224, 224]")
        dims = self.parse_shape_string(shape_str)
        
        if dims:
            return TensorShape(
                dims=dims,
                description="Batched data"
            )
        
        # Fallback
        batch_size = config.get("batch_size", 32)
        return TensorShape(
            dims=[batch_size, 3, 224, 224],
            description="Batched data"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # DataLoader is typically a source node, doesn't accept incoming connections
        return "DataLoader is a source node and cannot accept incoming connections"

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
            node_type='dataloader',
            node_id=node_id,
            init_params={},
            config_params=config,
            input_shape_info={'dims': []},
            output_shape_info={'dims': output_shape.dims if output_shape else []},
            template_context={}
        )
