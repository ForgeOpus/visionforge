/**
 * PyTorch Add (Element-wise) Node Definition
 * Enhanced with pattern-based validation and conflict reporting
 */

import { MergeNodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType, ShapePattern } from '../../../types'
import { addMerge } from '../../../validation/patterns'
import { getRank, isNumeric } from '../../../validation/matchers'

export class AddNode extends MergeNodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'add',
    label: 'Add',
    category: 'merge',
    color: 'var(--color-cyan)',
    icon: 'PlusCircle',
    description: 'Element-wise addition of tensors (same shape required)',
    framework: BackendFramework.PyTorch
  }

  /**
   * Input pattern: element-wise addition requires same shape
   */
  readonly inputPattern: ShapePattern = addMerge()

  readonly configSchema: ConfigField[] = []

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    // For add nodes, output shape matches input shape (all inputs must have same shape)
    return inputShape
  }

  /**
   * Special method for computing output shape with multiple inputs
   * All inputs must have the same shape for element-wise addition
   */
  computeMultiInputShape(inputShapes: TensorShape[], config: BlockConfig): TensorShape | undefined {
    if (inputShapes.length === 0) {
      return undefined
    }

    if (inputShapes.length === 1) {
      return inputShapes[0]
    }

    const firstShape = inputShapes[0]
    const rank = getRank(firstShape)

    // Validate all shapes have same rank
    const allSameRank = inputShapes.every(shape => getRank(shape) === rank)
    if (!allSameRank) {
      return undefined
    }

    // Validate all dimensions match
    for (let i = 0; i < rank; i++) {
      const dims = inputShapes.map(s => s.dims[i])
      const numericDims = dims.filter(isNumeric) as number[]

      if (numericDims.length > 1) {
        const unique = new Set(numericDims)
        if (unique.size > 1) {
          return undefined
        }
      }
    }

    // All shapes match, return the first shape with metadata
    return {
      dims: [...firstShape.dims],
      description: `Element-wise sum of ${inputShapes.length} tensors`,
      flags: {
        inferred: true
      },
      provenance: {
        source: 'computed',
        transformation: 'add',
        description: `Added ${inputShapes.length} tensors element-wise`
      }
    }
  }

  /**
   * Validate multiple inputs have matching shapes
   */
  validateMultipleInputs(inputShapes: TensorShape[], config: BlockConfig): string | undefined {
    if (inputShapes.length < 2) {
      return undefined
    }

    const firstShape = inputShapes[0]
    const rank = getRank(firstShape)

    // Check all shapes have same rank
    for (let i = 1; i < inputShapes.length; i++) {
      const inputRank = getRank(inputShapes[i])
      if (inputRank !== rank) {
        return `Add: rank mismatch - input 1 is ${rank}D, input ${i + 1} is ${inputRank}D`
      }
    }

    // Check all dimensions match
    const conflicts: string[] = []
    for (let d = 0; d < rank; d++) {
      const dims = inputShapes.map(s => s.dims[d])
      const numericDims = dims.filter(isNumeric) as number[]

      if (numericDims.length > 1) {
        const unique = new Set(numericDims)
        if (unique.size > 1) {
          conflicts.push(`dim ${d}: ${dims.join(' vs ')}`)
        }
      }
    }

    if (conflicts.length > 0) {
      return `Add: dimension mismatch (${conflicts.join('; ')})`
    }

    return undefined
  }

  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined {
    // Add nodes accept connections from any source
    // Shape matching is validated when multiple inputs are present
    return undefined
  }

  getDefaultConfig(): BlockConfig {
    return {}
  }
}
