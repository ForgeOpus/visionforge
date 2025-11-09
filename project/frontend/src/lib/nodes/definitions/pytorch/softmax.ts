/**
 * PyTorch Softmax Activation Node Definition
 */

import { PassthroughNodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField } from '../../../types'

export class SoftmaxNode extends PassthroughNodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'softmax',
    label: 'Softmax',
    category: 'activation',
    color: 'var(--color-destructive)',
    icon: 'Percent',
    description: 'Softmax activation',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'dim',
      label: 'Dimension',
      type: 'number',
      default: -1,
      description: 'Dimension along which to apply'
    }
  ]
}
