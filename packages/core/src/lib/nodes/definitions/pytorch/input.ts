/**
 * PyTorch Input Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'

export class InputNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'input',
    label: 'Input',
    category: 'input',
    color: 'var(--color-gray)',
    icon: 'CircleDashed',
    description: 'Input placeholder node',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'shape',
      label: 'Input Shape',
      type: 'text',
      default: '[1, 3, 224, 224]',
      placeholder: '[batch, channels, height, width]',
      description: 'Input tensor dimensions as JSON array. Overridden if connected to DataLoader.'
    },
    {
      name: 'label',
      label: 'Custom Label',
      type: 'text',
      default: 'Input',
      placeholder: 'Enter custom label...',
      description: 'Custom label for this input node'
    },
    {
      name: 'note',
      label: 'Note',
      type: 'text',
      placeholder: 'Add notes here...',
      description: 'Notes or comments about this input'
    }
  ]

  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined {
    // Input nodes can only receive connections from data source nodes
    const allowedSources: BlockType[] = ['dataloader']
    
    if (!allowedSources.includes(sourceNodeType)) {
      return `Input blocks can only receive connections from data source nodes (DataLoader). Got connection from ${sourceNodeType}`
    }
    
    // Connection is valid
    return undefined
  }

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    // If connected to a data source (DataLoader), use its shape
    if (inputShape) {
      return inputShape
    }
    
    // Otherwise, use manually configured shape
    if (config.shape && typeof config.shape === 'string') {
      const dims = this.parseShapeString(config.shape)
      if (dims) {
        return { dims }
      }
    }
    
    // Fallback to default shape
    const defaultDims = this.parseShapeString('[1, 3, 224, 224]')
    return defaultDims ? { dims: defaultDims } : undefined
  }
}
