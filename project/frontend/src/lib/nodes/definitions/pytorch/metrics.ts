/**
 * PyTorch Metrics Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'
import { PortDefinition } from '../../ports'

export class MetricsNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'metrics',
    label: 'Metrics',
    category: 'output',
    color: 'var(--color-success)',
    icon: 'BarChart3',
    description: 'Track multiple evaluation metrics during training',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'task_type',
      label: 'Task Type',
      type: 'select',
      default: 'binary_classification',
      required: true,
      options: [
        { value: 'binary_classification', label: 'Binary Classification' },
        { value: 'multiclass_classification', label: 'Multiclass Classification' },
        { value: 'multilabel_classification', label: 'Multilabel Classification' },
        { value: 'regression', label: 'Regression' }
      ],
      description: 'Type of task for metric selection'
    },
    {
      name: 'metrics',
      label: 'Metrics',
      type: 'multiselect',
      default: ['accuracy'],
      required: true,
      options: [
        // Classification metrics
        { value: 'accuracy', label: 'Accuracy' },
        { value: 'precision', label: 'Precision' },
        { value: 'recall', label: 'Recall' },
        { value: 'f1', label: 'F1 Score' },
        { value: 'specificity', label: 'Specificity' },
        { value: 'auroc', label: 'AUROC' },
        { value: 'auprc', label: 'AUPRC' },
        // Regression metrics
        { value: 'mse', label: 'Mean Squared Error' },
        { value: 'mae', label: 'Mean Absolute Error' },
        { value: 'rmse', label: 'Root Mean Squared Error' },
        { value: 'r2', label: 'R² Score' }
      ],
      description: 'Select one or more metrics to track during training'
    },
    {
      name: 'num_classes',
      label: 'Number of Classes',
      type: 'number',
      default: 2,
      min: 2,
      description: 'Required for multiclass classification'
    },
    {
      name: 'average',
      label: 'Averaging Method',
      type: 'select',
      default: 'macro',
      options: [
        { value: 'macro', label: 'Macro' },
        { value: 'micro', label: 'Micro' },
        { value: 'weighted', label: 'Weighted' },
        { value: 'none', label: 'None' }
      ],
      description: 'Averaging method for multi-class metrics'
    }
  ]

  /**
   * Get input ports based on task type
   */
  getInputPorts(config: BlockConfig): PortDefinition[] {
    return [
      {
        id: 'metrics-input-predictions',
        label: 'Predictions',
        type: 'input',
        semantic: 'predictions',
        required: true,
        description: 'Model predictions'
      },
      {
        id: 'metrics-input-targets',
        label: 'Targets',
        type: 'input',
        semantic: 'labels',
        required: true,
        description: 'Ground truth targets'
      }
    ]
  }

  /**
   * Metrics nodes are terminal nodes - they don't have output ports
   */
  getOutputPorts(config: BlockConfig): PortDefinition[] {
    return []
  }

  /**
   * Metrics node accepts multiple inputs
   */
  allowsMultipleInputs(): boolean {
    return true
  }

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    return { dims: [1], description: 'Metric value' }
  }

  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined {
    // Metrics node accepts any input shape
    return undefined
  }

  validateConfig(config: BlockConfig): string[] {
    const errors = super.validateConfig(config)

    // Validate metrics array
    const metrics = config.metrics
    if (!Array.isArray(metrics)) {
      errors.push('At least one metric is required')
    } else if (metrics.length === 0) {
      errors.push('At least one metric is required')
    }

    // Validate num_classes for multiclass tasks
    const taskType = config.task_type || 'binary_classification'
    if (taskType === 'multiclass_classification') {
      const numClasses = config.num_classes
      if (numClasses === undefined || numClasses < 2) {
        errors.push('Number of classes must be >= 2 for multiclass classification')
      }
    }

    return errors
  }
}
