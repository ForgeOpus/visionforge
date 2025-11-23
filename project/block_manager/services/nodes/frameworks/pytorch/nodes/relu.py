"""PyTorch ReLU Activation Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchPassthroughNode


class ReLUNode(PyTorchPassthroughNode):
    """Rectified Linear Unit activation"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="relu",
            label="ReLU",
            category="basic",
            color="var(--color-accent)",
            icon="Lightning",
            description="Rectified Linear Unit activation",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return []  # No configuration needed
