"""PyTorch Metrics Node Definition"""

from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework, LayerCodeSpec


class MetricsNode(NodeDefinition):
    """Metrics node for tracking multiple evaluation metrics during training"""

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="metrics",
            label="Metrics",
            category="output",
            color="var(--color-success)",
            icon="BarChart3",
            description="Track multiple evaluation metrics during training (OPTIONAL)",
            framework=Framework.PYTORCH
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="task_type",
                label="Task Type",
                type="select",
                default="binary_classification",
                required=True,
                options=[
                    {"value": "binary_classification", "label": "Binary Classification"},
                    {"value": "multiclass_classification", "label": "Multiclass Classification"},
                    {"value": "multilabel_classification", "label": "Multilabel Classification"},
                    {"value": "regression", "label": "Regression"}
                ],
                description="Type of task for metric selection"
            ),
            ConfigField(
                name="metrics",
                label="Metrics",
                type="multiselect",
                default=['accuracy'],
                required=True,
                options=[
                    {"value": "accuracy", "label": "Accuracy"},
                    {"value": "precision", "label": "Precision"},
                    {"value": "recall", "label": "Recall"},
                    {"value": "f1", "label": "F1 Score"},
                    {"value": "specificity", "label": "Specificity"},
                    {"value": "auroc", "label": "AUROC"},
                    {"value": "auprc", "label": "AUPRC"},
                    {"value": "mse", "label": "Mean Squared Error"},
                    {"value": "mae", "label": "Mean Absolute Error"},
                    {"value": "rmse", "label": "Root Mean Squared Error"},
                    {"value": "r2", "label": "R² Score"}
                ],
                description="Select one or more metrics to track during training"
            ),
            ConfigField(
                name="num_classes",
                label="Number of Classes",
                type="number",
                default=2,
                min=2,
                description="Required for multiclass classification, must be >= 2"
            ),
            ConfigField(
                name="average",
                label="Averaging Method",
                type="select",
                default="macro",
                options=[
                    {"value": "macro", "label": "Macro"},
                    {"value": "micro", "label": "Micro"},
                    {"value": "weighted", "label": "Weighted"},
                    {"value": "none", "label": "None"}
                ],
                description="Averaging method for multi-class metrics"
            )
        ]

    def compute_output_shape(
        self,
        _input_shape: Optional[TensorShape],
        _config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Metrics node outputs metric values (scalars)
        return TensorShape(
            dims=[1],
            description="Metric value"
        )

    def validate_incoming_connection(
        self,
        _source_node_type: str,
        _source_output_shape: Optional[TensorShape],
        _target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Metrics node accepts any input shape
        return None

    @property
    def allows_multiple_inputs(self) -> bool:
        """Metrics nodes accept multiple inputs (predictions, labels, etc.)"""
        return True

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate metrics configuration"""
        errors = super().validate_config(config)

        # Validate metrics array
        metrics = config.get('metrics', ['accuracy'])
        if not isinstance(metrics, list):
            errors.append("Metrics must be an array")
        elif len(metrics) == 0:
            errors.append("At least one metric is required")
        elif not all(isinstance(m, str) for m in metrics):
            errors.append("All metrics must be strings")

        # Validate num_classes for multiclass tasks
        task_type = config.get('task_type', 'binary_classification')
        if task_type == 'multiclass_classification':
            num_classes = config.get('num_classes')
            if num_classes is None or num_classes < 2:
                errors.append("Number of classes must be >= 2 for multiclass classification")

        return errors

    def get_pytorch_code_spec(
        self,
        node_id: str,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape],
        output_shape: Optional[TensorShape]
    ) -> LayerCodeSpec:
        """
        Metrics nodes don't generate layer code - they only provide configuration
        for the training script. This method exists for interface compatibility.
        """
        sanitized_id = node_id.replace('-', '_')

        return LayerCodeSpec(
            class_name='Metrics',
            layer_variable_name=f'{sanitized_id}_Metrics',
            node_type='metrics',
            node_id=node_id,
            init_params={},
            config_params=config,
            input_shape_info={'dims': input_shape.dims if input_shape else []},
            output_shape_info={'dims': [1]},
            template_context={}
        )
