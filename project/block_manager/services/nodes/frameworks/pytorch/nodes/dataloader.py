"""PyTorch DataLoader Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchSourceNode


class DataLoaderNode(PyTorchSourceNode):
    """Data loading and batching"""

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
                description="Shuffle data at every epoch"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # DataLoader doesn't define output shape itself
        # It will be defined by the dataset
        return None
