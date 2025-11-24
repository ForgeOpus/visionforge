"""PyTorch Dropout Layer Node Definition"""

from typing import List

from ....core.types import ConfigField, Framework, NodeMetadata
from ..base import PyTorchPassthroughNode


class DropoutNode(PyTorchPassthroughNode):
    """Dropout regularization layer"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="dropout",
            label="Dropout",
            category="basic",
            color="var(--color-accent)",
            icon="Minus",
            description="Dropout regularization",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="p",
                label="Dropout Probability",
                type="number",
                default=0.5,
                min=0.0,
                max=1.0,
                description="Probability of an element to be zeroed"
            )
        ]
