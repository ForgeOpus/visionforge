/**
 * PyTorch Output Node Definition
 */

import { TerminalNodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField } from '../../../types'
import { PortDefinition } from '../../ports'

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

  /**
   * Output node provides predictions that can connect to loss functions
   */
  getOutputPorts(config: BlockConfig): PortDefinition[] {
    return [{
      id: 'predictions-output',
      label: 'Predictions',
      type: 'output',
      semantic: 'predictions',
      required: false,
      description: 'Model predictions/output'
    }]
  }

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    return inputShape
  }
}
