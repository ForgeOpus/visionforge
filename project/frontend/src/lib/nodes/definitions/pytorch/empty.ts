/**
 * PyTorch Empty/Placeholder Node Definition
 */

import { PassthroughNodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField } from '../../../types'

export class EmptyNode extends PassthroughNodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'empty',
    label: 'Empty Node',
    category: 'utility',
    color: 'var(--color-gray)',
    icon: 'Placeholder',
    description: 'Placeholder node for architecture planning',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'label',
      label: 'Custom Label',
      type: 'text',
      default: 'Empty Node',
      placeholder: 'Enter custom label...',
      description: 'Custom label for this node'
    },
    {
      name: 'note',
      label: 'Note',
      type: 'text',
      placeholder: 'Add notes here...',
      description: 'Notes or comments about this placeholder'
    }
  ]
}
