/**
 * PyTorch BatchNorm Layer Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'

export class BatchNormNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'batchnorm',
    label: 'Batch Normalization',
    category: 'basic',
    color: 'var(--color-accent)',
    icon: 'ChartLineUp',
    description: 'Batch normalization layer',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'eps',
      label: 'Epsilon',
      type: 'number',
      default: 0.00001,
      min: 0,
      description: 'Value added to denominator for numerical stability'
    },
    {
      name: 'momentum',
      label: 'Momentum',
      type: 'number',
      default: 0.1,
      min: 0,
      max: 1,
      description: 'Momentum for running mean and variance'
    },
    {
      name: 'affine',
      label: 'Learnable Parameters',
      type: 'boolean',
      default: true,
      description: 'Enable learnable affine parameters'
    }
  ]

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    return inputShape
  }

  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined {
    // Allow connections from input/dataloader without shape validation
    if (sourceNodeType === 'input' || sourceNodeType === 'dataloader') {
      return undefined
    }

    // Empty and custom nodes are flexible
    if (sourceNodeType === 'empty' || sourceNodeType === 'custom') {
      return undefined
    }

    // Validate dimension requirement (2D or 4D)
    return this.validateDimensions(sourceOutputShape, {
      dims: [2, 4],
      description: '(2D for BatchNorm1d or 4D for BatchNorm2d)'
    })
  }
}
