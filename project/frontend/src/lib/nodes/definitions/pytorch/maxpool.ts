/**
 * PyTorch MaxPool2D Layer Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'

export class MaxPool2DNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'maxpool',
    label: 'MaxPool2D',
    category: 'basic',
    color: 'var(--color-purple)',
    icon: 'SquaresFour',
    description: '2D max pooling layer',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'kernel_size',
      label: 'Kernel Size',
      type: 'number',
      default: 2,
      min: 1,
      description: 'Size of pooling window'
    },
    {
      name: 'stride',
      label: 'Stride',
      type: 'number',
      default: 2,
      min: 1,
      description: 'Stride of pooling window'
    },
    {
      name: 'padding',
      label: 'Padding',
      type: 'number',
      default: 0,
      min: 0,
      description: 'Zero-padding added to both sides'
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
    const kernel = config.kernel_size as number
    const stride = config.stride as number
    const padding = config.padding as number

    const [outHeight, outWidth] = this.computePool2DOutput(
      height,
      width,
      kernel,
      stride,
      padding
    )

    return {
      dims: [batch, channels, outHeight, outWidth],
      description: 'Pooled feature map'
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
