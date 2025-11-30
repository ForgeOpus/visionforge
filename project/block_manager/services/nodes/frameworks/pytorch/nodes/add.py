"""PyTorch Add Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchMergeNode


class AddNode(PyTorchMergeNode):
    """Element-wise addition of tensors"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="add",
            label="Add",
            category="merge",
            color="var(--color-accent)",
            icon="Plus",
            description="Element-wise addition",
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
        # Add preserves shape (all inputs must match)
        if not input_shape:
            return None

        return TensorShape(
            dims=input_shape.dims,
            description="Sum of inputs"
        )
