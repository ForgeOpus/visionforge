/**
 * PyTorch AvgPool2D Layer Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'

export class AvgPool2DNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'avgpool2d',
    label: 'AvgPool2D',
    category: 'basic',
    color: 'var(--color-primary)',
    icon: 'ArrowsInSimple',
    description: 'Average pooling for 2D inputs',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'kernel_size',
      label: 'Kernel Size',
      type: 'number',
      default: 2,
      min: 1,
      description: 'Size of the pooling window'
    },
    {
      name: 'stride',
      label: 'Stride',
      type: 'number',
      default: 2,
      min: 1,
      description: 'Stride of the pooling window'
    },
    {
      name: 'padding',
      label: 'Padding',
      type: 'number',
      default: 0,
      min: 0,
      description: 'Zero padding on both sides'
    }
  ]

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    if (!inputShape) {
      return undefined
    }

    if (inputShape.dims.length !== 4) {
      return undefined
    }

    const [batch, channels, height, width] = inputShape.dims as number[]
    const kernel = (config.kernel_size ?? 2) as number
    const stride = (config.stride ?? 2) as number
    const padding = (config.padding ?? 0) as number

    const outHeight = Math.floor((height + 2 * padding - kernel) / stride) + 1
    const outWidth = Math.floor((width + 2 * padding - kernel) / stride) + 1

    return {
      dims: [batch, channels, outHeight, outWidth],
      description: `AvgPool2D(${kernel}x${kernel})`
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
      dims: 4,
      description: '[batch, channels, height, width]'
    })
  }
}
