/**
 * PyTorch Add (Element-wise) Node Definition
 */

import { MergeNodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'

export class AddNode extends MergeNodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'add',
    label: 'Add',
    category: 'merge',
    color: 'var(--color-cyan)',
    icon: 'PlusCircle',
    description: 'Element-wise addition of tensors',
    framework: BackendFramework.PyTorch
  }

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
    
    // Validate all shapes match exactly
    const allMatch = inputShapes.every(shape => this.shapesMatch(firstShape, shape))
    
    if (!allMatch) {
      return undefined
    }

    // All shapes match, return the first shape
    return {
      dims: firstShape.dims,
      description: `Element-wise sum of ${inputShapes.length} tensors`
    }
  }

  /**
   * Special method for validating multiple inputs have matching shapes
   */
  validateMultipleInputs(inputShapes: TensorShape[]): string | undefined {
    if (inputShapes.length < 2) {
      return undefined
    }

    const firstShape = inputShapes[0]
    
    // All shapes must match exactly
    for (let i = 1; i < inputShapes.length; i++) {
      if (!this.shapesMatch(firstShape, inputShapes[i])) {
        return `Add requires all inputs to have the same shape. First input: [${firstShape.dims.join(', ')}], Input ${i + 1}: [${inputShapes[i].dims.join(', ')}]`
      }
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
}
