"""TensorFlow DataLoader Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework


class DataLoaderNode(NodeDefinition):
    """DataLoader for feeding data to the network using tf.keras.utils.PyDataset"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="dataloader",
            label="DataLoader",
            category="input",
            color="var(--color-teal)",
            icon="Database",
            description="Data loading and batching (TensorFlow PyDataset)",
            framework=Framework.TENSORFLOW
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
                name="output_shape",
                label="Output Shape",
                type="string",
                default="[32, 224, 224, 3]",
                description="Shape of batched data in NHWC format (e.g., [batch, height, width, channels])"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Parse output shape from config
        shape_str = config.get("output_shape", "[32, 224, 224, 3]")
        dims = self.parse_shape_string(shape_str)
        
        if dims:
            return TensorShape(
                dims=dims,
                description="Batched data (NHWC)"
            )
        
        # Fallback
        batch_size = config.get("batch_size", 32)
        return TensorShape(
            dims=[batch_size, 224, 224, 3],
            description="Batched data (NHWC)"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # DataLoader is typically a source node, doesn't accept incoming connections
        return "DataLoader is a source node and cannot accept incoming connections"
