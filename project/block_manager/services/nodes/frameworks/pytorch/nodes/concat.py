"""PyTorch Concat Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchMergeNode


class ConcatNode(PyTorchMergeNode):
    """Concatenate tensors along a dimension"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="concat",
            label="Concatenate",
            category="merge",
            color="var(--color-accent)",
            icon="GitMerge",
            description="Concatenate tensors along a dimension",
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
                description="Dimension along which to concatenate"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # For concat, output shape depends on all inputs
        # This simplified version assumes single input for shape display
        if not input_shape:
            return None

        # Return input shape as placeholder
        # Actual shape computation needs all inputs
        return TensorShape(
            dims=input_shape.dims,
            description="Concatenated tensor"
        )
