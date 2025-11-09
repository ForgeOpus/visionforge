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

  readonly configSchema: ConfigField[] = [
    {
      name: 'output_type',
      label: 'Output Type',
      type: 'select',
      default: 'classification',
      options: [
        { value: 'classification', label: 'Classification' },
        { value: 'regression', label: 'Regression' },
        { value: 'segmentation', label: 'Segmentation' },
        { value: 'custom', label: 'Custom' }
      ],
      description: 'Type of model output'
    },
    {
      name: 'num_classes',
      label: 'Number of Classes',
      type: 'number',
      default: 10,
      min: 1,
      description: 'Number of output classes (for classification)'
    }
  ]

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    return inputShape
  }
}
