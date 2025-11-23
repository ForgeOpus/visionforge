"""PyTorch Softmax Activation Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchPassthroughNode


class SoftmaxNode(PyTorchPassthroughNode):
    """Softmax activation function"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="softmax",
            label="Softmax",
            category="basic",
            color="var(--color-accent)",
            icon="Function",
            description="Softmax activation function",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="dim",
                label="Dimension",
                type="number",
                default=1,
                description="Dimension along which Softmax will be computed"
            )
        ]
