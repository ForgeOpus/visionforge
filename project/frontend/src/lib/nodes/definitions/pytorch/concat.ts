/**
 * PyTorch Concatenate Node Definition
 */

import { MergeNodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField } from '../../../types'

export class ConcatNode extends MergeNodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'concat',
    label: 'Concatenate',
    category: 'merge',
    color: 'var(--color-cyan)',
    icon: 'GitBranch',
    description: 'Concatenate multiple tensors',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'dim',
      label: 'Concatenation Dimension',
      type: 'number',
      default: 1,
      description: 'Dimension along which to concatenate (typically channel dimension)'
    }
  ]

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    // Note: For concat nodes with multiple inputs, the full shape computation
    // needs to be handled in the store which has access to all input edges
    // This method handles the single-input case
    return inputShape
  }

  /**
   * Special method for computing output shape with multiple inputs
   * This should be called by the store/registry when multiple inputs are available
   */
  computeMultiInputShape(inputShapes: TensorShape[], config: BlockConfig): TensorShape | undefined {
    if (inputShapes.length === 0) {
      return undefined
    }

    if (inputShapes.length === 1) {
      return inputShapes[0]
    }

    const dim = (config.dim as number) || 1
    const firstShape = inputShapes[0]

    // Validate all shapes have same number of dimensions
    const allSameDims = inputShapes.every(shape => shape.dims.length === firstShape.dims.length)
    if (!allSameDims) {
      return undefined
    }

    // Validate all dimensions match except the concat dimension
    for (let i = 0; i < firstShape.dims.length; i++) {
      if (i === dim) continue
      const allMatch = inputShapes.every(shape => shape.dims[i] === firstShape.dims[i])
      if (!allMatch) {
        return undefined
      }
    }

    // Compute concatenated dimension size
    const concatDimSize = inputShapes.reduce((sum, shape) => {
      return sum + (shape.dims[dim] as number)
    }, 0)

    // Build output shape
    const outputDims = [...firstShape.dims]
    outputDims[dim] = concatDimSize

    return {
      dims: outputDims,
      description: `Concatenated along dim ${dim}`
    }
  }
}
