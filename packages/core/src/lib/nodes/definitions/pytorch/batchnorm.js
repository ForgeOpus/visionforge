/**
 * PyTorch BatchNorm Layer Node Definition
 * Enhanced with pattern-based validation
 */
import { NodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
import { rankOneOf, passThrough } from '../../../validation/patterns';
import { getRank } from '../../../validation/matchers';
export class BatchNormNode extends NodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'batchnorm',
                label: 'Batch Normalization',
                category: 'basic',
                color: 'var(--color-accent)',
                icon: 'ChartLineUp',
                description: 'Batch normalization layer (2D or 4D input)',
                framework: BackendFramework.PyTorch
            }
        });
        /**
         * Input pattern: 2D for BatchNorm1d or 4D for BatchNorm2d
         */
        Object.defineProperty(this, "inputPattern", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: rankOneOf([2, 3, 4])
        });
        /**
         * Output pattern: same as input (pass-through)
         */
        Object.defineProperty(this, "outputPattern", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: passThrough()
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [
                {
                    name: 'eps',
                    label: 'Epsilon',
                    type: 'number',
                    default: 0.00001,
                    min: 0,
                    description: 'Value added to denominator for numerical stability'
                },
                {
                    name: 'momentum',
                    label: 'Momentum',
                    type: 'number',
                    default: 0.1,
                    min: 0,
                    max: 1,
                    description: 'Momentum for running mean and variance'
                },
                {
                    name: 'affine',
                    label: 'Learnable Parameters',
                    type: 'boolean',
                    default: true,
                    description: 'Enable learnable affine parameters'
                }
            ]
        });
    }
    computeOutputShape(inputShape, config) {
        if (!inputShape) {
            return undefined;
        }
        // Pass through with metadata
        return {
            dims: [...inputShape.dims],
            description: 'Normalized output',
            flags: { inferred: true },
            provenance: {
                source: 'computed',
                transformation: 'batchnorm'
            }
        };
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
        if (!sourceOutputShape) {
            return undefined;
        }
        const rank = getRank(sourceOutputShape);
        if (rank !== 2 && rank !== 3 && rank !== 4) {
            return `BatchNorm requires 2D, 3D, or 4D input, got ${rank}D`;
        }
        return undefined;
    }
    getDefaultConfig() {
        return {
            eps: 0.00001,
            momentum: 0.1,
            affine: true
        };
    }
}
