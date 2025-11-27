"""TensorFlow Empty/Placeholder Node Definition"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowNodeDefinition


class EmptyNode(TensorFlowNodeDefinition):
    """
    Placeholder node for architecture planning.

    Passes input through unchanged.
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="empty",
            label="Empty",
            category="utility",
            color="var(--color-gray)",
            icon="Circle",
            description="Placeholder for architecture planning",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="note",
                label="Note",
                type="text",
                description="Notes about this placeholder"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        return input_shape

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return None
