/**
 * PyTorch ReLU Activation Node Definition
 */

import { PassthroughNodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField } from '../../../types'

export class ReLUNode extends PassthroughNodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'relu',
    label: 'ReLU',
    category: 'basic',
    color: 'var(--color-accent)',
    icon: 'Pulse',
    description: 'ReLU activation function',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'inplace',
      label: 'In-place Operation',
      type: 'boolean',
      default: false,
      description: 'Perform operation in-place to save memory'
    }
  ]
}
