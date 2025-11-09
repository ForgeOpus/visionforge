/**
 * PyTorch Conv2D Layer Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'

export class Conv2DNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'conv2d',
    label: 'Conv2D',
    category: 'basic',
    color: 'var(--color-purple)',
    icon: 'SquareHalf',
    description: '2D convolutional layer',
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
      description: 'Size of convolving kernel'
    },
    {
      name: 'stride',
      label: 'Stride',
      type: 'number',
      default: 1,
      min: 1,
      description: 'Stride of convolution'
    },
    {
      name: 'padding',
      label: 'Padding',
      type: 'number',
      default: 0,
      min: 0,
      description: 'Zero-padding added to both sides'
    },
    {
      name: 'dilation',
      label: 'Dilation',
      type: 'number',
      default: 1,
      min: 1,
      description: 'Spacing between kernel elements'
    }
  ]

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    if (!inputShape || !config.out_channels) {
      return undefined
    }

    if (inputShape.dims.length !== 4) {
      return undefined
    }

    const [batch, _, height, width] = inputShape.dims as number[]
    const kernel = config.kernel_size as number
    const stride = config.stride as number
    const padding = config.padding as number
    const dilation = config.dilation as number

    const [outHeight, outWidth] = this.computeConv2DOutput(
      height,
      width,
      kernel,
      stride,
      padding,
      dilation
    )

    return {
      dims: [batch, config.out_channels as number, outHeight, outWidth],
      description: 'Convolved feature map'
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
      dims: 4,
      description: '[batch, channels, height, width]'
    })
  }
}
