/**
 * PyTorch Dropout Layer Node Definition
 */

import { PassthroughNodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField } from '../../../types'

export class DropoutNode extends PassthroughNodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'dropout',
    label: 'Dropout',
    category: 'basic',
    color: 'var(--color-accent)',
    icon: 'Drop',
    description: 'Dropout regularization layer',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'p',
      label: 'Drop Probability',
      type: 'number',
      default: 0.5,
      min: 0,
      max: 1,
      description: 'Probability of an element being zeroed'
    },
    {
      name: 'inplace',
      label: 'In-place Operation',
      type: 'boolean',
      default: false,
      description: 'Perform operation in-place to save memory'
    }
  ]

  validateConfig(config: BlockConfig): string[] {
    const errors = super.validateConfig(config)

    const p = config.p as number
    if (p !== undefined && (p < 0 || p > 1)) {
      errors.push('Drop Probability must be between 0 and 1')
    }

    return errors
  }
}
