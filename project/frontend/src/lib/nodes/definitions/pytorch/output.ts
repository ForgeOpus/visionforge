/**
 * PyTorch Output Node Definition
 */

import { TerminalNodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField } from '../../../types'

export class OutputNode extends TerminalNodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'output',
    label: 'Output',
    category: 'output',
    color: 'var(--color-green)',
    icon: 'Export',
    description: 'Define model output and predictions',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = []

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    return inputShape
  }
}
