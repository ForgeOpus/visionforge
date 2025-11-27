/**
 * PyTorch Multi-Head Attention Node Definition
 */
import { NodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
export class AttentionNode extends NodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'attention',
                label: 'Multi-Head Attention',
                category: 'advanced',
                color: 'var(--color-purple)',
                icon: 'Brain',
                description: 'Multi-head self-attention mechanism',
                framework: BackendFramework.PyTorch
            }
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [
                {
                    name: 'embed_dim',
                    label: 'Embedding Dimension',
                    type: 'number',
                    required: true,
                    min: 1,
                    description: 'Total dimension of the model'
                },
                {
                    name: 'num_heads',
                    label: 'Number of Heads',
                    type: 'number',
                    default: 8,
                    min: 1,
                    description: 'Number of parallel attention heads'
                },
                {
                    name: 'dropout',
                    label: 'Dropout',
                    type: 'number',
                    default: 0.0,
                    min: 0,
                    max: 1,
                    description: 'Dropout probability on attention weights'
                },
                {
                    name: 'bias',
                    label: 'Use Bias',
                    type: 'boolean',
                    default: true,
                    description: 'Add bias to input/output projection layers'
                }
            ]
        });
    }
    computeOutputShape(inputShape, config) {
        // Multi-head attention maintains input shape [batch, sequence, embedding]
        return inputShape;
    }
    validateIncomingConnection(sourceNodeType, sourceOutputShape, targetConfig) {
        // Allow connections from input/dataloader without shape validation
        if (sourceNodeType === 'input' || sourceNodeType === 'dataloader') {
            return undefined;
        }
        // Empty and custom nodes are flexible
        if (sourceNodeType === 'empty' || sourceNodeType === 'custom') {
            return undefined;
        }
        // Validate dimension requirement
        return this.validateDimensions(sourceOutputShape, {
            dims: 3,
            description: '[batch, sequence, embedding]'
        });
    }
    validateConfig(config) {
        const errors = super.validateConfig(config);
        // Validate that embed_dim is divisible by num_heads
        const embedDim = config.embed_dim;
        const numHeads = config.num_heads;
        if (embedDim && numHeads && embedDim % numHeads !== 0) {
            errors.push('Embedding Dimension must be divisible by Number of Heads');
        }
        return errors;
    }
}
