/**
 * PyTorch Flatten Layer Node Definition
 * Enhanced with pattern-based validation
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, ShapePattern, DimensionValue } from '../../../types'
import { minRank } from '../../../validation/patterns'
import { getRank, isNumeric } from '../../../validation/matchers'

export class FlattenNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'flatten',
    label: 'Flatten',
    category: 'basic',
    color: 'var(--color-primary)',
    icon: 'ArrowsDownUp',
    description: 'Flatten tensor dimensions',
    framework: BackendFramework.PyTorch
  }

  /**
   * Input pattern: at least 2D tensor
   */
  readonly inputPattern: ShapePattern = minRank(2)

  readonly configSchema: ConfigField[] = [
    {
      name: 'start_dim',
      label: 'Start Dimension',
      type: 'number',
      default: 1,
      min: 0,
      description: 'First dimension to flatten'
    },
    {
      name: 'end_dim',
      label: 'End Dimension',
      type: 'number',
      default: -1,
      description: 'Last dimension to flatten (-1 for last)'
    }
  ]

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    if (!inputShape) {
      return undefined
    }

    const rank = getRank(inputShape)
    const startDim = (config.start_dim as number) ?? 1
    let endDim = (config.end_dim as number) ?? -1

    // Normalize negative index
    if (endDim < 0) {
      endDim = rank + endDim
    }

    // Validate dimensions
    if (startDim < 0 || startDim >= rank || endDim < startDim || endDim >= rank) {
      return inputShape
    }

    // Get dimensions to flatten
    const dimsToFlatten = inputShape.dims.slice(startDim, endDim + 1)

    // Calculate flattened size
    let flattenedSize: DimensionValue
    if (dimsToFlatten.every(isNumeric)) {
      flattenedSize = (dimsToFlatten as number[]).reduce((a, b) => a * b, 1)
    } else {
      // Symbolic - create expression
      flattenedSize = dimsToFlatten.map(d => String(d)).join('*')
    }

    // Build output shape
    const outputDims: DimensionValue[] = [
      ...inputShape.dims.slice(0, startDim),
      flattenedSize,
      ...inputShape.dims.slice(endDim + 1)
    ]

    return {
      dims: outputDims,
      description: `Flattened dims ${startDim} to ${endDim}`,
      flags: {
        inferred: true,
        symbolic: !isNumeric(flattenedSize)
      },
      provenance: {
        source: 'computed',
        transformation: 'flatten',
        description: `${inputShape.dims.join('×')} → ${outputDims.join('×')}`
      }
    }
  }

  getDefaultConfig(): BlockConfig {
    return {
      start_dim: 1,
      end_dim: -1
    }
  }
}
