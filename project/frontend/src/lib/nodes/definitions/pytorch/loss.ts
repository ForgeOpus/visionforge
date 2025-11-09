/**
 * PyTorch Loss Function Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'
import { PortDefinition } from '../../ports'

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
  getInputPorts(config: BlockConfig): PortDefinition[] {
    const lossType = config.loss_type || 'cross_entropy'

    const portConfigs: Record<string, PortDefinition[]> = {
      cross_entropy: [
        { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Model predictions' },
        { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'True labels' }
      ],
      mse: [
        { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Model predictions' },
        { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'True values' }
      ],
      mae: [
        { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Model predictions' },
        { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'True values' }
      ],
      bce: [
        { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Model predictions' },
        { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'True labels' }
      ],
      nll: [
        { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Model predictions' },
        { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'True labels' }
      ],
      smooth_l1: [
        { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Model predictions' },
        { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'True values' }
      ],
      kl_div: [
        { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Predicted distribution' },
        { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'Target distribution' }
      ],
      triplet: [
        { id: 'loss-input-anchor', label: 'Anchor', type: 'input', semantic: 'anchor', required: true, description: 'Anchor embedding' },
        { id: 'loss-input-positive', label: 'Positive', type: 'input', semantic: 'positive', required: true, description: 'Positive example embedding' },
        { id: 'loss-input-negative', label: 'Negative', type: 'input', semantic: 'negative', required: true, description: 'Negative example embedding' }
      ],
      contrastive: [
        { id: 'loss-input-input1', label: 'Input 1', type: 'input', semantic: 'input1', required: true, description: 'First input embedding' },
        { id: 'loss-input-input2', label: 'Input 2', type: 'input', semantic: 'input2', required: true, description: 'Second input embedding' },
        { id: 'loss-input-label', label: 'Label', type: 'input', semantic: 'labels', required: true, description: 'Similarity label (1 or -1)' }
      ],
      custom: [
        { id: 'loss-input-input1', label: 'Input 1', type: 'input', semantic: 'input1', required: true, description: 'First input' },
        { id: 'loss-input-input2', label: 'Input 2', type: 'input', semantic: 'input2', required: true, description: 'Second input' }
      ]
    }

    return portConfigs[lossType as string] || portConfigs.cross_entropy
  }
  
  /**
   * Get output ports - loss always outputs a single scalar loss value
   */
  getOutputPorts(config: BlockConfig): PortDefinition[] {
    return [{
      id: 'loss-output',
      label: 'Loss',
      type: 'output',
      semantic: 'loss',
      required: false,
      description: 'Scalar loss value'
    }]
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
