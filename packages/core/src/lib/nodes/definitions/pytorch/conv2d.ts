/**
 * PyTorch Conv2D Layer Node Definition
 * Enhanced with pattern-based validation
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType, ShapePattern, DimensionValue } from '../../../types'
import { spatialInput } from '../../../validation/patterns'
import { getRank, isNumeric } from '../../../validation/matchers'

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

  /**
   * Input pattern: 4D spatial tensor (B, C, H, W)
   */
  readonly inputPattern: ShapePattern = spatialInput()

  readonly configSchema: ConfigField[] = [
    {
      name: 'in_channels',
      label: 'Input Channels',
      type: 'number',
      required: false,
      min: 1,
      description: 'Number of input channels (auto-inferred if not set)'
    },
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

    const rank = getRank(inputShape)
    if (rank !== 4) {
      return undefined
    }

    const [batch, , height, width] = inputShape.dims
    const kernel = (config.kernel_size as number) || 3
    const stride = (config.stride as number) || 1
    const padding = (config.padding as number) || 0
    const dilation = (config.dilation as number) || 1
    const outChannels = config.out_channels as number

    // Compute output dimensions
    let outHeight: DimensionValue
    let outWidth: DimensionValue

    if (isNumeric(height)) {
      const [h] = this.computeConv2DOutput(height, 1, kernel, stride, padding, dilation)
      outHeight = h
    } else {
      outHeight = `${height}'`
    }

    if (isNumeric(width)) {
      const [, w] = this.computeConv2DOutput(1, width, kernel, stride, padding, dilation)
      outWidth = w
    } else {
      outWidth = `${width}'`
    }

    return {
      dims: [batch, outChannels, outHeight, outWidth],
      description: 'Convolved feature map',
      flags: {
        inferred: true,
        symbolic: !isNumeric(outHeight) || !isNumeric(outWidth)
      },
      provenance: {
        source: 'computed',
        transformation: 'conv2d'
      }
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

    if (!sourceOutputShape) {
      return undefined
    }

    const rank = getRank(sourceOutputShape)
    if (rank !== 4) {
      return `Conv2D requires 4D input (B, C, H, W), got ${rank}D`
    }

    // Check input channels if specified
    const inChannels = targetConfig.in_channels as number | undefined
    if (inChannels && inChannels > 0) {
      const inputChannels = sourceOutputShape.dims[1]
      if (isNumeric(inputChannels) && inputChannels !== inChannels) {
        return `Channel mismatch: input has ${inputChannels} channels, expected ${inChannels}`
      }
    }

    return undefined
  }

  getDefaultConfig(): BlockConfig {
    return {
      out_channels: '',
      kernel_size: 3,
      stride: 1,
      padding: 0,
      dilation: 1
    }
  }
}
