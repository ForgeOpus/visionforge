/**
 * PyTorch Flatten Layer Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField } from '../../../types'

export class FlattenNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'flatten',
    label: 'Flatten',
    category: 'basic',
    color: 'var(--color-primary)',
    icon: 'ArrowsDownUp',
    description: 'Flatten tensor to 2D',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'start_dim',
      label: 'Start Dimension',
      type: 'number',
      default: 1,
      min: 0,
      description: 'First dimension to flatten'
    }
  ]

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    if (!inputShape) {
      return undefined
    }

    const startDim = (config.start_dim as number) || 1
    const dims = inputShape.dims as number[]

    if (startDim >= dims.length) {
      return inputShape
    }

    // Calculate flattened size
    let flattenedSize = 1
    for (let i = startDim; i < dims.length; i++) {
      flattenedSize *= dims[i]
    }

    // Keep dimensions before start_dim, flatten the rest
    const outputDims = dims.slice(0, startDim)
    outputDims.push(flattenedSize)

    return {
      dims: outputDims,
      description: 'Flattened tensor'
    }
  }
}
