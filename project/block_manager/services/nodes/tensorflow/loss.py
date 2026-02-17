"""TensorFlow Loss Function Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class LossNode(NodeDefinition):
    """Loss function node for defining training loss"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="loss",
            label="Loss Function",
            category="output",
            color="var(--color-destructive)",
            icon="Target",
            description="Define loss function for training (REQUIRED for code export)",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="loss_type",
                label="Loss Type",
                type="select",
                default="cross_entropy",
                required=True,
                options=[
                    {"value": "cross_entropy", "label": "Sparse Categorical Cross Entropy"},
                    {"value": "mse", "label": "Mean Squared Error"},
                    {"value": "mae", "label": "Mean Absolute Error"},
                    {"value": "bce", "label": "Binary Cross Entropy"},
                    {"value": "categorical_crossentropy", "label": "Categorical Cross Entropy"},
                    {"value": "kl_div", "label": "KL Divergence"},
                    {"value": "hinge", "label": "Hinge Loss"}
                ],
                description="Type of loss function to use for training"
            ),
            ConfigField(
                name="reduction",
                label="Reduction",
                type="select",
                default="sum_over_batch_size",
                options=[
                    {"value": "sum_over_batch_size", "label": "Sum Over Batch Size (Default)"},
                    {"value": "sum", "label": "Sum"},
                    {"value": "none", "label": "None"}
                ],
                description="How to reduce the loss across the batch"
            ),
            ConfigField(
                name="from_logits",
                label="From Logits",
                type="boolean",
                default=True,
                description="Whether predictions are logits (True) or probabilities (False)"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Loss node outputs a scalar value
        return TensorShape(
            dims=[1],
            description="Scalar loss value"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Loss node accepts any input shape
        return None

    @property
    def allows_multiple_inputs(self) -> bool:
        """Loss nodes accept multiple inputs (predictions, labels, etc.)"""
        return True

    def get_tensorflow_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """
        Loss nodes don't generate layer code - they only provide configuration
        for the training script. This method exists for interface compatibility.
        """
        sanitized_id = node_id.replace('-', '_')

        return LayerCodeSpec(
            class_name='Loss',
            layer_variable_name=f'{sanitized_id}_Loss',
            node_type='loss',
            node_id=node_id,
            init_params={},
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': [1]},
            template_context={}
        )
