/**
 * PyTorch Ground Truth Node Definition
 */

import { SourceNodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField } from '../../../types'
import { PortDefinition } from '../../ports'

export class GroundTruthNode extends SourceNodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'groundtruth',
    label: 'Ground Truth',
    category: 'input',
    color: 'var(--color-orange)',
    icon: 'Target',
    description: 'Ground truth labels for training',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'shape',
      label: 'Label Shape',
      type: 'text',
      default: '[1, 10]',
      required: true,
      placeholder: '[batch, num_classes]',
      description: 'Ground truth tensor dimensions as JSON array'
    },
    {
      name: 'label',
      label: 'Custom Label',
      type: 'text',
      default: 'Ground Truth',
      placeholder: 'Enter custom label...',
      description: 'Custom label for this ground truth node'
    },
    {
      name: 'note',
      label: 'Note',
      type: 'text',
      placeholder: 'Add notes here...',
      description: 'Notes or comments about this ground truth data'
    }
  ]

  /**
   * Ground truth outputs labels, not data
   */
  getOutputPorts(config: BlockConfig): PortDefinition[] {
    return [{
      id: 'default',
      label: 'Labels',
      type: 'output',
      semantic: 'labels',
      required: false,
      description: 'Ground truth labels for training'
    }]
  }

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    const shapeStr = String(config.shape || '[1, 10]')
    const dims = this.parseShapeString(shapeStr)

    if (dims) {
      return {
        dims,
        description: 'Ground truth labels'
      }
    }

    return undefined
  }

  validateConfig(config: BlockConfig): string[] {
    const errors = super.validateConfig(config)

    // Validate shape format
    const shapeStr = String(config.shape || '')
    const dims = this.parseShapeString(shapeStr)
    if (!dims) {
      errors.push('Label Shape must be a valid JSON array of positive numbers')
    }

    return errors
  }
}
