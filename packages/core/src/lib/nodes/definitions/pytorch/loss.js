/**
 * PyTorch Loss Function Node Definition
 * Enhanced with dual-input validation for prediction-target compatibility
 */
import { NodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
import { wildcard, scalarOutput } from '../../../validation/patterns';
import { getRank, isNumeric } from '../../../validation/matchers';
export class LossNode extends NodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'loss',
                label: 'Loss Function',
                category: 'output',
                color: 'var(--color-destructive)',
                icon: 'Target',
                description: 'Define loss function for training with shape validation',
                framework: BackendFramework.PyTorch
            }
        });
        /**
         * Input pattern: accepts any shape (validated based on loss type)
         */
        Object.defineProperty(this, "inputPattern", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: wildcard()
        });
        /**
         * Output pattern: scalar loss value
         */
        Object.defineProperty(this, "outputPattern", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: scalarOutput()
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [
                {
                    name: 'loss_type',
                    label: 'Loss Type',
                    type: 'select',
                    default: 'cross_entropy',
                    required: true,
                    options: [
                        { value: 'cross_entropy', label: 'Cross Entropy Loss' },
                        { value: 'mse', label: 'Mean Squared Error' },
                        { value: 'mae', label: 'Mean Absolute Error' },
                        { value: 'bce', label: 'Binary Cross Entropy' },
                        { value: 'nll', label: 'Negative Log Likelihood' },
                        { value: 'smooth_l1', label: 'Smooth L1 Loss' },
                        { value: 'kl_div', label: 'KL Divergence' },
                        { value: 'triplet', label: 'Triplet Loss' },
                        { value: 'contrastive', label: 'Contrastive Loss' },
                        { value: 'custom', label: 'Custom Loss' }
                    ],
                    description: 'Type of loss function to use'
                },
                {
                    name: 'reduction',
                    label: 'Reduction',
                    type: 'select',
                    default: 'mean',
                    options: [
                        { value: 'mean', label: 'Mean' },
                        { value: 'sum', label: 'Sum' },
                        { value: 'none', label: 'None' }
                    ],
                    description: 'How to reduce the loss'
                },
                {
                    name: 'weight',
                    label: 'Class Weights',
                    type: 'text',
                    placeholder: '[1.0, 1.0, 2.0, ...]',
                    description: 'Optional class weights as JSON array'
                }
            ]
        });
    }
    /**
     * Get input ports based on the loss type configuration
     */
    getInputPorts(config) {
        const lossType = config.loss_type || 'cross_entropy';
        const portConfigs = {
            cross_entropy: [
                { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Logits (B, K) or (B, T, K)' },
                { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'Class indices (B,) or (B, T)' }
            ],
            mse: [
                { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Model predictions (same shape as target)' },
                { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'True values (same shape as predictions)' }
            ],
            mae: [
                { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Model predictions (same shape as target)' },
                { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'True values (same shape as predictions)' }
            ],
            bce: [
                { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Logits or probabilities' },
                { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'Binary labels (same shape)' }
            ],
            nll: [
                { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Log probabilities (B, K)' },
                { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'Class indices (B,)' }
            ],
            smooth_l1: [
                { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Model predictions' },
                { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'True values' }
            ],
            kl_div: [
                { id: 'loss-input-y_pred', label: 'Predictions', type: 'input', semantic: 'predictions', required: true, description: 'Log probabilities' },
                { id: 'loss-input-y_true', label: 'Ground Truth', type: 'input', semantic: 'labels', required: true, description: 'Target distribution' }
            ],
            triplet: [
                { id: 'loss-input-anchor', label: 'Anchor', type: 'input', semantic: 'anchor', required: true, description: 'Anchor embedding (B, F)' },
                { id: 'loss-input-positive', label: 'Positive', type: 'input', semantic: 'positive', required: true, description: 'Positive example (B, F)' },
                { id: 'loss-input-negative', label: 'Negative', type: 'input', semantic: 'negative', required: true, description: 'Negative example (B, F)' }
            ],
            contrastive: [
                { id: 'loss-input-input1', label: 'Input 1', type: 'input', semantic: 'input1', required: true, description: 'First embedding (B, F)' },
                { id: 'loss-input-input2', label: 'Input 2', type: 'input', semantic: 'input2', required: true, description: 'Second embedding (B, F)' },
                { id: 'loss-input-label', label: 'Label', type: 'input', semantic: 'labels', required: true, description: 'Similarity label' }
            ],
            custom: [
                { id: 'loss-input-input1', label: 'Input 1', type: 'input', semantic: 'input1', required: true, description: 'First input' },
                { id: 'loss-input-input2', label: 'Input 2', type: 'input', semantic: 'input2', required: true, description: 'Second input' }
            ]
        };
        return portConfigs[lossType] || portConfigs.cross_entropy;
    }
    /**
     * Get output ports - loss always outputs a single scalar loss value
     */
    getOutputPorts(config) {
        return [{
                id: 'loss-output',
                label: 'Loss',
                type: 'output',
                semantic: 'loss',
                required: false,
                description: 'Scalar loss value'
            }];
    }
    /**
     * Loss node accepts multiple inputs but always outputs a scalar loss
     */
    allowsMultipleInputs() {
        return true;
    }
    computeOutputShape(inputShape, config) {
        const reduction = config.reduction || 'mean';
        // If reduction is 'none', output shape depends on input
        if (reduction === 'none' && inputShape) {
            return {
                dims: [...inputShape.dims],
                description: 'Unreduced loss (per-element)',
                flags: { inferred: true },
                provenance: {
                    source: 'computed',
                    transformation: 'loss'
                }
            };
        }
        // Otherwise, scalar output
        return {
            dims: [1],
            description: 'Scalar loss',
            flags: { inferred: true },
            provenance: {
                source: 'computed',
                transformation: 'loss'
            }
        };
    }
    validateIncomingConnection(sourceNodeType, sourceOutputShape, targetConfig) {
        // Allow connections from input/dataloader without shape validation
        if (sourceNodeType === 'input' || sourceNodeType === 'dataloader') {
            return undefined;
        }
        // If no shape, allow connection (will validate later with both inputs)
        if (!sourceOutputShape) {
            return undefined;
        }
        return undefined;
    }
    /**
     * Validate that prediction and target shapes are compatible for the loss type
     */
    validatePredictionTarget(predictionShape, targetShape, lossType) {
        const predRank = getRank(predictionShape);
        const targetRank = getRank(targetShape);
        switch (lossType) {
            case 'cross_entropy':
            case 'nll':
                // Prediction: (B, K) or (B, T, K), Target: (B,) or (B, T)
                if (predRank === 2) {
                    // Prediction (B, K), target should be (B,)
                    if (targetRank !== 1) {
                        return `CrossEntropy: with 2D predictions (B, K), target should be 1D (B,), got ${targetRank}D`;
                    }
                    // Check batch dimensions match
                    if (isNumeric(predictionShape.dims[0]) && isNumeric(targetShape.dims[0])) {
                        if (predictionShape.dims[0] !== targetShape.dims[0]) {
                            return `Batch size mismatch: predictions ${predictionShape.dims[0]}, target ${targetShape.dims[0]}`;
                        }
                    }
                }
                else if (predRank === 3) {
                    // Prediction (B, T, K), target should be (B, T)
                    if (targetRank !== 2) {
                        return `CrossEntropy: with 3D predictions (B, T, K), target should be 2D (B, T), got ${targetRank}D`;
                    }
                    // Check batch and sequence dimensions match
                    for (let i = 0; i < 2; i++) {
                        if (isNumeric(predictionShape.dims[i]) && isNumeric(targetShape.dims[i])) {
                            if (predictionShape.dims[i] !== targetShape.dims[i]) {
                                return `Dimension ${i} mismatch: predictions ${predictionShape.dims[i]}, target ${targetShape.dims[i]}`;
                            }
                        }
                    }
                }
                else {
                    return `CrossEntropy expects 2D or 3D predictions, got ${predRank}D`;
                }
                break;
            case 'mse':
            case 'mae':
            case 'smooth_l1':
                // Same shape required
                if (predRank !== targetRank) {
                    return `${lossType.toUpperCase()}: prediction and target must have same rank, got ${predRank}D vs ${targetRank}D`;
                }
                // Check all dimensions match
                for (let i = 0; i < predRank; i++) {
                    if (isNumeric(predictionShape.dims[i]) && isNumeric(targetShape.dims[i])) {
                        if (predictionShape.dims[i] !== targetShape.dims[i]) {
                            return `${lossType.toUpperCase()}: dimension ${i} mismatch: ${predictionShape.dims[i]} vs ${targetShape.dims[i]}`;
                        }
                    }
                }
                break;
            case 'bce':
                // Same shape or broadcastable
                if (predRank !== targetRank) {
                    // Allow broadcasting for BCE
                    if (targetRank !== 1 || predRank !== 2) {
                        return `BCE: incompatible shapes - predictions ${predRank}D, target ${targetRank}D`;
                    }
                }
                else {
                    // Same rank - check dimensions
                    for (let i = 0; i < predRank; i++) {
                        if (isNumeric(predictionShape.dims[i]) && isNumeric(targetShape.dims[i])) {
                            if (predictionShape.dims[i] !== targetShape.dims[i]) {
                                return `BCE: dimension ${i} mismatch: ${predictionShape.dims[i]} vs ${targetShape.dims[i]}`;
                            }
                        }
                    }
                }
                break;
            case 'kl_div':
                // Same shape required
                if (predRank !== targetRank) {
                    return `KL Divergence: shapes must match, got ${predRank}D vs ${targetRank}D`;
                }
                break;
            case 'triplet':
            case 'contrastive':
                // All inputs should have same shape (checked separately)
                break;
            default:
                // Custom loss - no specific validation
                break;
        }
        return undefined;
    }
    /**
     * Validate triplet loss inputs (anchor, positive, negative must match)
     */
    validateTripletInputs(anchor, positive, negative) {
        const anchorRank = getRank(anchor);
        const positiveRank = getRank(positive);
        const negativeRank = getRank(negative);
        if (anchorRank !== positiveRank || anchorRank !== negativeRank) {
            return `Triplet loss: all inputs must have same rank (${anchorRank}D, ${positiveRank}D, ${negativeRank}D)`;
        }
        // Check all dimensions match
        for (let i = 0; i < anchorRank; i++) {
            const dims = [anchor.dims[i], positive.dims[i], negative.dims[i]];
            const numericDims = dims.filter(isNumeric);
            if (numericDims.length > 1) {
                const unique = new Set(numericDims);
                if (unique.size > 1) {
                    return `Triplet loss: dimension ${i} mismatch - anchor: ${anchor.dims[i]}, positive: ${positive.dims[i]}, negative: ${negative.dims[i]}`;
                }
            }
        }
        return undefined;
    }
    validateConfig(config) {
        const errors = super.validateConfig(config);
        // Validate weight format if provided
        if (config.weight && config.weight !== '') {
            const weightStr = String(config.weight);
            try {
                const weights = JSON.parse(weightStr);
                if (!Array.isArray(weights) || !weights.every(w => typeof w === 'number')) {
                    errors.push('Class Weights must be a JSON array of numbers');
                }
            }
            catch {
                errors.push('Class Weights must be valid JSON format');
            }
        }
        return errors;
    }
    getDefaultConfig() {
        return {
            loss_type: 'cross_entropy',
            reduction: 'mean',
            weight: ''
        };
    }
}
