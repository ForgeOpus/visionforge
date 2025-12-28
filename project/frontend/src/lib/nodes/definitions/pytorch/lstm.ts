/**
 * PyTorch LSTM Layer Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'

export class LSTMNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'lstm',
    label: 'LSTM',
    category: 'advanced',
    color: 'var(--color-purple)',
    icon: 'ArrowsClockwise',
    description: 'LSTM recurrent layer',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'hidden_size',
      label: 'Hidden Size',
      type: 'number',
      required: true,
      min: 1,
      description: 'Number of features in hidden state'
    },
    {
      name: 'num_layers',
      label: 'Layers',
      type: 'number',
      default: 1,
      min: 1,
      description: 'Number of recurrent layers'
    },
    {
      name: 'bias',
      label: 'Use Bias',
      type: 'boolean',
      default: true,
      description: 'Use bias weights'
    },
    {
      name: 'batch_first',
      label: 'Batch First',
      type: 'boolean',
      default: true,
      description: 'Input shape is (batch, seq, feature)'
    },
    {
      name: 'dropout',
      label: 'Dropout',
      type: 'number',
      default: 0.0,
      min: 0.0,
      max: 1.0,
      description: 'Dropout probability (if layers > 1)'
    },
    {
      name: 'bidirectional',
      label: 'Bidirectional',
      type: 'boolean',
      default: false,
      description: 'Use bidirectional LSTM'
    }
  ]

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    if (!inputShape || !config.hidden_size) {
      return undefined
    }

    if (inputShape.dims.length !== 3) {
      return undefined
    }

    const batchFirst = config.batch_first ?? true
    const hiddenSize = config.hidden_size as number
    const bidirectional = config.bidirectional ?? false

    let batch: number, seqLen: number

    if (batchFirst) {
      [batch, seqLen] = inputShape.dims as number[]
    } else {
      [seqLen, batch] = inputShape.dims as number[]
    }

    const outFeatures = hiddenSize * (bidirectional ? 2 : 1)

    const dims = batchFirst ? [batch, seqLen, outFeatures] : [seqLen, batch, outFeatures]

    return {
      dims,
      description: `LSTM(${hiddenSize})`
    }
  }

  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined {
    if (sourceNodeType === 'input' || sourceNodeType === 'dataloader') {
      return undefined
    }

    if (sourceNodeType === 'empty' || sourceNodeType === 'custom') {
      return undefined
    }

    return this.validateDimensions(sourceOutputShape, {
      dims: 3,
      description: '[batch, sequence, features] or [sequence, batch, features]'
    })
  }
}
