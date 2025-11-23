"""PyTorch Output Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchTerminalNode


class OutputNode(PyTorchTerminalNode):
    """Output layer for the neural network"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="output",
            label="Output",
            category="output",
            color="var(--color-green)",
            icon="Export",
            description="Define model output and predictions",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return []  # No configuration needed

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Output shape is same as input
        return input_shape
