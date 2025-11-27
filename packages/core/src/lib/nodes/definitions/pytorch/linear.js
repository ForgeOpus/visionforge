/**
 * PyTorch Linear Layer Node Definition
 * Enhanced with pattern-based validation and auto_flatten support
 */
import { NodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
import { projectionInput } from '../../../validation/patterns';
import { getRank, isNumeric, getLastDim } from '../../../validation/matchers';
export class LinearNode extends NodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'linear',
                label: 'Linear',
                category: 'basic',
                color: 'var(--color-primary)',
                icon: 'Lightning',
                description: 'Fully connected layer (supports rank ≥2 with auto-flatten)',
                framework: BackendFramework.PyTorch
            }
        });
        /**
         * Input pattern: accepts any rank ≥2 with features as last dimension
         */
        Object.defineProperty(this, "inputPattern", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: projectionInput()
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [
                {
                    name: 'in_features',
                    label: 'Input Features',
                    type: 'number',
                    required: false,
                    min: 1,
                    description: 'Number of input features (auto-inferred if not set)'
                },
                {
                    name: 'out_features',
                    label: 'Output Features',
                    type: 'number',
                    required: true,
                    min: 1,
                    description: 'Number of output features'
                },
                {
                    name: 'bias',
                    label: 'Use Bias',
                    type: 'boolean',
                    default: true,
                    description: 'Add learnable bias'
                },
                {
                    name: 'auto_flatten',
                    label: 'Auto Flatten',
                    type: 'boolean',
                    default: false,
                    description: 'Automatically flatten input to 2D before projection'
                }
            ]
        });
    }
    computeOutputShape(inputShape, config) {
        // Check if out_features is properly set
        const outFeatures = config.out_features;
        if (!inputShape || outFeatures === undefined || outFeatures === null || outFeatures === '') {
            return undefined;
        }
        const numOutFeatures = Number(outFeatures);
        if (isNaN(numOutFeatures) || numOutFeatures <= 0) {
            return undefined;
        }
        const rank = getRank(inputShape);
        // Must be at least 2D
        if (rank < 2) {
            return undefined;
        }
        const autoFlatten = config.auto_flatten === true;
        // If auto_flatten is enabled and rank > 2, flatten to 2D
        if (autoFlatten && rank > 2) {
            // Flatten all dimensions except batch into features
            const batchDim = inputShape.dims[0];
            // Compute flattened feature dimension
            const featureDims = inputShape.dims.slice(1);
            let flattenedFeatures;
            if (featureDims.every(isNumeric)) {
                flattenedFeatures = featureDims.reduce((a, b) => a * b, 1);
            }
            else {
                // Symbolic - create expression
                flattenedFeatures = featureDims.map(d => String(d)).join('*');
            }
            return {
                dims: [batchDim, numOutFeatures],
                description: `Linear output (auto-flattened from ${rank}D)`,
                flags: { autoFlatten: true, inferred: true },
                provenance: {
                    source: 'computed',
                    transformation: 'linear',
                    description: `Flattened ${inputShape.dims.join('×')} → (${batchDim}, ${flattenedFeatures}) → (${batchDim}, ${numOutFeatures})`
                }
            };
        }
        // Standard behavior: preserve all leading dimensions, replace last with out_features
        const outputDims = [
            ...inputShape.dims.slice(0, -1),
            numOutFeatures
        ];
        return {
            dims: outputDims,
            description: rank === 2
                ? 'Linear output'
                : `Linear output (${rank}D preserved)`,
            flags: { inferred: true },
            provenance: {
                source: 'computed',
                transformation: 'linear'
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
        // If no shape provided, allow connection (will validate later)
        if (!sourceOutputShape) {
            return undefined;
        }
        const rank = getRank(sourceOutputShape);
        // Must be at least 2D
        if (rank < 2) {
            return `Linear requires at least 2D input, got ${rank}D. Add Reshape or Flatten.`;
        }
        // Check feature dimension compatibility if in_features is specified
        const inFeatures = targetConfig.in_features;
        if (inFeatures && inFeatures > 0) {
            const lastDim = getLastDim(sourceOutputShape);
            if (lastDim !== undefined && isNumeric(lastDim) && lastDim !== inFeatures) {
                // If auto_flatten is enabled and rank > 2, calculate flattened features
                if (targetConfig.auto_flatten === true && rank > 2) {
                    const featureDims = sourceOutputShape.dims.slice(1);
                    if (featureDims.every(isNumeric)) {
                        const flattenedFeatures = featureDims.reduce((a, b) => a * b, 1);
                        if (flattenedFeatures !== inFeatures) {
                            return `After auto-flatten: ${flattenedFeatures} features, expected ${inFeatures}`;
                        }
                        // Flattened features match - OK
                        return undefined;
                    }
                }
                return `Feature dimension mismatch: got ${lastDim}, expected ${inFeatures}`;
            }
        }
        return undefined;
    }
    /**
     * Validate configuration with feature inference
     */
    validateConfig(config) {
        const errors = [];
        // Check out_features
        if (config.out_features === undefined || config.out_features === '') {
            errors.push('Output Features is required');
        }
        else {
            const outFeatures = Number(config.out_features);
            if (isNaN(outFeatures) || outFeatures < 1) {
                errors.push('Output Features must be a positive number');
            }
        }
        // Validate in_features if provided
        if (config.in_features !== undefined && config.in_features !== '') {
            const inFeatures = Number(config.in_features);
            if (isNaN(inFeatures) || inFeatures < 1) {
                errors.push('Input Features must be a positive number');
            }
        }
        return errors;
    }
    getDefaultConfig() {
        return {
            out_features: '',
            bias: true,
            auto_flatten: false
        };
    }
}
