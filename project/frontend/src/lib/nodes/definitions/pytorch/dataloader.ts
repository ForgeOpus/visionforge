/**
 * PyTorch DataLoader Node Definition
 */

import { SourceNodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField } from '../../../types'

export class DataLoaderNode extends SourceNodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'dataloader',
    label: 'Data Loader',
    category: 'input',
    color: 'var(--color-teal)',
    icon: 'Database',
    description: 'Load and prepare input data with optional ground truth',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'shape',
      label: 'Input Shape',
      type: 'text',
      default: '[1, 3, 224, 224]',
      required: true,
      placeholder: '[batch, channels, height, width]',
      description: 'Input tensor dimensions as JSON array'
    },
    {
      name: 'has_ground_truth',
      label: 'Include Ground Truth Output',
      type: 'boolean',
      default: false,
      description: 'Enable a second output for ground truth labels'
    },
    {
      name: 'ground_truth_shape',
      label: 'Ground Truth Shape',
      type: 'text',
      default: '[1, 10]',
      placeholder: '[batch, num_classes]',
      description: 'Shape for ground truth labels (used when ground truth is enabled)'
    },
    {
      name: 'randomize',
      label: 'Randomize Data',
      type: 'boolean',
      default: false,
      description: 'Use random synthetic data for testing'
    },
    {
      name: 'csv_file',
      label: 'CSV File Path',
      type: 'text',
      placeholder: 'data/dataset.csv',
      description: 'Path to CSV file for data loading (optional)'
    }
  ]

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    const shapeStr = String(config.shape || '[1]')
    const dims = this.parseShapeString(shapeStr)

    if (dims) {
      return {
        dims,
        description: 'Input tensor'
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
      errors.push('Input Shape must be a valid JSON array of positive numbers')
    }

    // Validate ground truth shape if enabled
    if (config.has_ground_truth) {
      const gtShapeStr = String(config.ground_truth_shape || '')
      const gtDims = this.parseShapeString(gtShapeStr)
      if (!gtDims) {
        errors.push('Ground Truth Shape must be a valid JSON array when ground truth is enabled')
      }
    }

    return errors
  }
}
