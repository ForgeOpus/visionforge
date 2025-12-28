/**
 * PyTorch AdaptiveAvgPool2D Layer Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'

export class AdaptiveAvgPool2DNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'adaptiveavgpool2d',
    label: 'AdaptiveAvgPool2D',
    category: 'basic',
    color: 'var(--color-primary)',
    icon: 'Resize',
    description: 'Adaptive average pooling to fixed output size',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'output_size',
      label: 'Output Size',
      type: 'text',
      default: '1',
      description: 'Target output size (single number or [H, W])'
    }
  ]

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    if (!inputShape) {
      return undefined
    }

    if (inputShape.dims.length !== 4) {
      return undefined
    }

    const [batch, channels] = inputShape.dims as number[]
    const outputSizeStr = String(config.output_size ?? '1')

    let outHeight: number, outWidth: number

    try {
      if (outputSizeStr.includes(',') || outputSizeStr.includes('[')) {
        const cleaned = outputSizeStr.replace(/[\[\]\(\)\s]/g, '')
        const parts = cleaned.split(',')
        outHeight = parseInt(parts[0])
        outWidth = parts.length > 1 ? parseInt(parts[1]) : outHeight
      } else {
        outHeight = outWidth = parseInt(outputSizeStr)
      }
    } catch {
      outHeight = outWidth = 1
    }

    return {
      dims: [batch, channels, outHeight, outWidth],
      description: `AdaptiveAvgPool2D(${outHeight}x${outWidth})`
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
