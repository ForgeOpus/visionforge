/**
 * PyTorch Loss Function Node Definition
 */

import { TerminalNodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField } from '../../../types'

export class LossNode extends TerminalNodeDefinition {
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

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    return { dims: [1], description: 'Scalar loss' }
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
