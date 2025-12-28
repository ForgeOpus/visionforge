/**
 * PyTorch Conv1D Layer Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'

export class Conv1DNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'conv1d',
    label: 'Conv1D',
    category: 'advanced',
    color: 'var(--color-purple)',
    icon: 'WaveSquare',
    description: '1D convolutional layer',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'out_channels',
      label: 'Output Channels',
      type: 'number',
      required: true,
      min: 1,
      description: 'Number of output channels'
    },
    {
      name: 'kernel_size',
      label: 'Kernel Size',
      type: 'number',
      default: 3,
      min: 1,
      description: 'Size of the convolving kernel'
    },
    {
      name: 'stride',
      label: 'Stride',
      type: 'number',
      default: 1,
      min: 1,
      description: 'Stride of the convolution'
    },
    {
      name: 'padding',
      label: 'Padding',
      type: 'number',
      default: 0,
      min: 0,
      description: 'Zero padding on both sides'
    },
    {
      name: 'dilation',
      label: 'Dilation',
      type: 'number',
      default: 1,
      min: 1,
      description: 'Spacing between kernel elements'
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
    if (!inputShape || !config.out_channels) {
      return undefined
    }

    if (inputShape.dims.length !== 3) {
      return undefined
    }

    const [batch, _, length] = inputShape.dims as number[]

    const kernel = (config.kernel_size ?? 3) as number
    const stride = (config.stride ?? 1) as number
    const padding = (config.padding ?? 0) as number
    const dilation = (config.dilation ?? 1) as number

    const outLength = Math.floor((length + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1

    return {
      dims: [batch, config.out_channels as number, outLength],
      description: `Conv1D(${config.out_channels})`
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
      description: '[batch, channels, length]'
    })
  }
}
