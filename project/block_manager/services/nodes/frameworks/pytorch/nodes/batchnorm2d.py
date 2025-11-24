"""PyTorch BatchNorm2D Layer Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchPassthroughNode


class BatchNorm2DNode(PyTorchPassthroughNode):
    """Batch Normalization for 2D inputs"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="batchnorm",
            label="BatchNorm2D",
            category="basic",
            color="var(--color-accent)",
            icon="ChartLineUp",
            description="Batch normalization layer",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="eps",
                label="Epsilon",
                type="number",
                default=1e-5,
                min=0,
                description="Value added to denominator for numerical stability"
            ),
            ConfigField(
                name="momentum",
                label="Momentum",
                type="number",
                default=0.1,
                min=0,
                max=1,
                description="Momentum for running mean and variance"
            ),
            ConfigField(
                name="affine",
                label="Learnable Parameters",
                type="boolean",
                default=True,
                description="Enable learnable affine parameters"
            )
        ]

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        if source_node_type in ("input", "dataloader", "empty", "custom"):
            return None

        return self.validate_dimensions(
            source_output_shape,
            4,
            "[batch, channels, height, width]"
        )
