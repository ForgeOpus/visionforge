/**
 * PyTorch Linear Layer Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'

export class LinearNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'linear',
    label: 'Linear',
    category: 'basic',
    color: 'var(--color-primary)',
    icon: 'Lightning',
    description: 'Fully connected layer',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'out_features',
      label: 'Output Features',
      type: 'number',
      required: true,
      min: 1,
      description: 'Number of output features'
    },
    {
      name: 'bias',
      label: 'Use Bias',
      type: 'boolean',
      default: true,
      description: 'Add learnable bias'
    }
  ]

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    // Check if out_features is properly set (not undefined, null, or empty)
    const outFeatures = config.out_features
    if (!inputShape || outFeatures === undefined || outFeatures === null || outFeatures === '') {
      return undefined
    }

    if (inputShape.dims.length !== 2) {
      return undefined
    }

    const numOutFeatures = Number(outFeatures)
    if (isNaN(numOutFeatures) || numOutFeatures <= 0) {
      return undefined
    }

    return {
      dims: [inputShape.dims[0], numOutFeatures],
      description: 'Fully connected output'
    }
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

    // Validate dimension requirement
    return this.validateDimensions(sourceOutputShape, {
      dims: 2,
      description: '[batch, features]'
    })
  }
}
