"""TensorFlow DataLoader Node"""

from typing import Any, Dict, List, Optional

from ....core.types import ConfigField, ConfigOption, Framework, NodeMetadata, TensorShape
from ..base import TensorFlowSourceNode


class DataLoaderNode(TensorFlowSourceNode):
    """
    Data Loader/Dataset Configuration.

    Defines data loading parameters using tf.data.Dataset.
    TensorFlow uses NHWC format.

    Output: [batch_size, height, width, channels] or [batch_size, features]
    """

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="dataloader",
            label="DataLoader",
            category="input",
            color="var(--color-teal)",
            icon="Database",
            description="Data loading configuration",
            framework=Framework.TENSORFLOW
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="dataset",
                label="Dataset",
                type="select",
                default="MNIST",
                options=[
                    ConfigOption(value="MNIST", label="MNIST"),
                    ConfigOption(value="CIFAR10", label="CIFAR-10"),
                    ConfigOption(value="CIFAR100", label="CIFAR-100"),
                    ConfigOption(value="ImageNet", label="ImageNet"),
                    ConfigOption(value="Custom", label="Custom")
                ],
                description="Dataset to load"
            ),
            ConfigField(
                name="batch_size",
                label="Batch Size",
                type="number",
                default=32,
                min=1,
                description="Number of samples per batch"
            ),
            ConfigField(
                name="shuffle",
                label="Shuffle",
                type="boolean",
                default=True,
                description="Shuffle data each epoch"
            )
        ]

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        dataset = config.get("dataset", "MNIST")
        batch_size = int(config.get("batch_size", 32))

        # NHWC format for TensorFlow
        dataset_shapes = {
            "MNIST": [batch_size, 28, 28, 1],
            "CIFAR10": [batch_size, 32, 32, 3],
            "CIFAR100": [batch_size, 32, 32, 3],
            "ImageNet": [batch_size, 224, 224, 3],
            "Custom": [batch_size, 224, 224, 3]
        }

        dims = dataset_shapes.get(dataset, [batch_size, 224, 224, 3])

        return TensorShape(
            dims=dims,
            description=f"{dataset} batch (NHWC)"
        )

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return "DataLoader cannot have incoming connections"
