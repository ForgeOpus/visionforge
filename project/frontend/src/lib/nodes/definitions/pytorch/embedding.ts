/**
 * PyTorch Embedding Layer Node Definition
 */

import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types'

export class EmbeddingNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'embedding',
    label: 'Embedding',
    category: 'advanced',
    color: 'var(--color-purple)',
    icon: 'TextAa',
    description: 'Token embedding layer',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'num_embeddings',
      label: 'Vocabulary Size',
      type: 'number',
      required: true,
      min: 1,
      description: 'Size of the vocabulary'
    },
    {
      name: 'embedding_dim',
      label: 'Embedding Dimension',
      type: 'number',
      required: true,
      min: 1,
      description: 'Size of each embedding vector'
    },
    {
      name: 'padding_idx',
      label: 'Padding Index',
      type: 'number',
      default: -1,
      description: 'Padding token index (or -1 for none)'
    },
    {
      name: 'max_norm',
      label: 'Max Norm',
      type: 'number',
      default: 0,
      min: 0,
      description: 'Renormalize embeddings (0 for no normalization)'
    },
    {
      name: 'scale_grad_by_freq',
      label: 'Scale Grad by Freq',
      type: 'boolean',
      default: false,
      description: 'Scale gradients by word frequency'
    }
  ]

  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    if (!inputShape || !config.embedding_dim) {
      return undefined
    }

    const embeddingDim = config.embedding_dim as number

    if (inputShape.dims.length === 2) {
      const [batch, seqLen] = inputShape.dims as number[]
      return {
        dims: [batch, seqLen, embeddingDim],
        description: `Embedding(${embeddingDim})`
      }
    }

    return {
      dims: [...inputShape.dims, embeddingDim],
      description: `Embedding(${embeddingDim})`
    }
  }

  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined {
    if (sourceNodeType === 'input' || sourceNodeType === 'dataloader') {
      return undefined
    }

    if (sourceNodeType === 'empty' || sourceNodeType === 'custom') {
      return undefined
    }

    return undefined
  }
}
