"""PyTorch Linear (Fully Connected) Layer Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import PyTorchNodeDefinition


class LinearNode(PyTorchNodeDefinition):
    """
    Linear/Fully Connected Layer.

    Applies a linear transformation: y = xW^T + b

    Input: [batch_size, in_features]
    Output: [batch_size, out_features]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="linear",
            label="Linear",
            category="basic",
            color="var(--color-primary)",
            icon="Lightning",
            description="Fully connected layer",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="out_features",
                label="Output Features",
                type="number",
                required=True,
                min=1,
                description="Number of output features"
            ),
            ConfigField(
                name="bias",
                label="Use Bias",
                type="boolean",
                default=True,
                description="Add learnable bias"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Check if out_features is properly set
        out_features = config.get("out_features")
        if not input_shape or out_features is None or out_features == '':
            return None

        if len(input_shape.dims) != 2:
            return None

        try:
            num_out_features = int(out_features)
            if num_out_features <= 0:
                return None
        except (ValueError, TypeError):
            return None

        return TensorShape(
            dims=[input_shape.dims[0], num_out_features],
            description="Fully connected output"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Allow connections from input/dataloader without shape validation
        if source_node_type in ("input", "dataloader"):
            return None

        # Empty and custom nodes are flexible
        if source_node_type in ("empty", "custom"):
            return None

        # Validate dimension requirement
        return self.validate_dimensions(
            source_output_shape,
            2,
            "[batch, features]"
        )
