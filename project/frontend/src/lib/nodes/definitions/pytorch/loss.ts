/**
 * PyTorch Loss Function Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'

export interface InputPort {
  id: string
  label: string
  description: string
}

export class LossNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'loss',
    label: 'Loss Function',
    category: 'output',
    color: 'var(--color-destructive)',
    icon: 'Target',
    description: 'Define loss function for training',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'loss_type',
      label: 'Loss Type',
      type: 'select',
      default: 'cross_entropy',
      required: true,
      options: [
        { value: 'cross_entropy', label: 'Cross Entropy Loss' },
        { value: 'mse', label: 'Mean Squared Error' },
        { value: 'mae', label: 'Mean Absolute Error' },
        { value: 'bce', label: 'Binary Cross Entropy' },
        { value: 'nll', label: 'Negative Log Likelihood' },
        { value: 'smooth_l1', label: 'Smooth L1 Loss' },
        { value: 'kl_div', label: 'KL Divergence' },
        { value: 'triplet', label: 'Triplet Loss' },
        { value: 'contrastive', label: 'Contrastive Loss' },
        { value: 'custom', label: 'Custom Loss' }
      ],
      description: 'Type of loss function to use'
    },
    {
      name: 'reduction',
      label: 'Reduction',
      type: 'select',
      default: 'mean',
      options: [
        { value: 'mean', label: 'Mean' },
        { value: 'sum', label: 'Sum' },
        { value: 'none', label: 'None' }
      ],
      description: 'How to reduce the loss'
    },
    {
      name: 'weight',
      label: 'Class Weights',
      type: 'text',
      placeholder: '[1.0, 1.0, 2.0, ...]',
      description: 'Optional class weights as JSON array'
    }
  ]

  /**
   * Get input ports based on the loss type configuration
   */
  getInputPorts(config: BlockConfig): InputPort[] {
    const lossType = config.loss_type || 'cross_entropy'

    const portConfigs: Record<string, InputPort[]> = {
      cross_entropy: [
        { id: 'y_pred', label: 'Predictions', description: 'Model predictions' },
        { id: 'y_true', label: 'Ground Truth', description: 'True labels' }
      ],
      mse: [
        { id: 'y_pred', label: 'Predictions', description: 'Model predictions' },
        { id: 'y_true', label: 'Ground Truth', description: 'True values' }
      ],
      mae: [
        { id: 'y_pred', label: 'Predictions', description: 'Model predictions' },
        { id: 'y_true', label: 'Ground Truth', description: 'True values' }
      ],
      bce: [
        { id: 'y_pred', label: 'Predictions', description: 'Model predictions' },
        { id: 'y_true', label: 'Ground Truth', description: 'True labels' }
      ],
      nll: [
        { id: 'y_pred', label: 'Predictions', description: 'Model predictions' },
        { id: 'y_true', label: 'Ground Truth', description: 'True labels' }
      ],
      smooth_l1: [
        { id: 'y_pred', label: 'Predictions', description: 'Model predictions' },
        { id: 'y_true', label: 'Ground Truth', description: 'True values' }
      ],
      kl_div: [
        { id: 'y_pred', label: 'Predictions', description: 'Predicted distribution' },
        { id: 'y_true', label: 'Ground Truth', description: 'Target distribution' }
      ],
      triplet: [
        { id: 'anchor', label: 'Anchor', description: 'Anchor embedding' },
        { id: 'positive', label: 'Positive', description: 'Positive example embedding' },
        { id: 'negative', label: 'Negative', description: 'Negative example embedding' }
      ],
      contrastive: [
        { id: 'input1', label: 'Input 1', description: 'First input embedding' },
        { id: 'input2', label: 'Input 2', description: 'Second input embedding' },
        { id: 'label', label: 'Label', description: 'Similarity label (1 or -1)' }
      ],
      custom: [
        { id: 'input1', label: 'Input 1', description: 'First input' },
        { id: 'input2', label: 'Input 2', description: 'Second input' }
      ]
    }

    return portConfigs[lossType as string] || portConfigs.cross_entropy
  }

  /**
   * Loss node accepts multiple inputs but always outputs a scalar loss
   */
  allowsMultipleInputs(): boolean {
    return true
  }

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    return { dims: [1], description: 'Scalar loss' }
  }

  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined {
    // Loss node accepts any input shape (predictions and labels)
    return undefined
  }

  validateConfig(config: BlockConfig): string[] {
    const errors = super.validateConfig(config)

    // Validate weight format if provided
    if (config.weight && config.weight !== '') {
      const weightStr = String(config.weight)
      try {
        const weights = JSON.parse(weightStr)
        if (!Array.isArray(weights) || !weights.every(w => typeof w === 'number')) {
          errors.push('Class Weights must be a JSON array of numbers')
        }
      } catch {
        errors.push('Class Weights must be valid JSON format')
      }
    }

    return errors
  }
}
