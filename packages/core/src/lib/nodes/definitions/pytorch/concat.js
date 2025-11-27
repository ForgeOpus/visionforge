/**
 * PyTorch Concatenate Node Definition
 * Enhanced with pattern-based validation and conflict reporting
 */
import { MergeNodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
import { concatMerge } from '../../../validation/patterns';
import { getRank, isNumeric } from '../../../validation/matchers';
export class ConcatNode extends MergeNodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'concat',
                label: 'Concatenate',
                category: 'merge',
                color: 'var(--color-cyan)',
                icon: 'GitBranch',
                description: 'Concatenate multiple tensors along specified dimension',
                framework: BackendFramework.PyTorch
            }
        });
        /**
         * Input pattern: concat merge along configurable axis
         */
        Object.defineProperty(this, "inputPattern", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: concatMerge(1)
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [
                {
                    name: 'dim',
                    label: 'Concatenation Dimension',
                    type: 'number',
                    default: 1,
                    min: 0,
                    description: 'Dimension along which to concatenate (typically channel dimension)'
                }
            ]
        });
    }
    computeOutputShape(inputShape, config) {
        // For concat nodes with multiple inputs, the full shape computation
        // needs to be handled in the store which has access to all input edges
        // This method handles the single-input case
        return inputShape;
    }
    /**
     * Special method for computing output shape with multiple inputs
     * This should be called by the store/registry when multiple inputs are available
     */
    computeMultiInputShape(inputShapes, config) {
        if (inputShapes.length === 0) {
            return undefined;
        }
        if (inputShapes.length === 1) {
            return inputShapes[0];
        }
        const dim = config.dim ?? 1;
        const firstShape = inputShapes[0];
        const rank = getRank(firstShape);
        // Validate dimension is in range
        if (dim < 0 || dim >= rank) {
            return undefined;
        }
        // Validate all shapes have same number of dimensions
        const allSameRank = inputShapes.every(shape => getRank(shape) === rank);
        if (!allSameRank) {
            return undefined;
        }
        // Validate all dimensions match except the concat dimension
        for (let i = 0; i < rank; i++) {
            if (i === dim)
                continue;
            // Collect dimensions at this index
            const dims = inputShapes.map(s => s.dims[i]);
            const numericDims = dims.filter(isNumeric);
            // If we have multiple numeric values, they must match
            if (numericDims.length > 1) {
                const unique = new Set(numericDims);
                if (unique.size > 1) {
                    return undefined;
                }
            }
        }
        // Compute concatenated dimension size
        let concatDimSize = 0;
        let hasSymbolic = false;
        for (const shape of inputShapes) {
            const d = shape.dims[dim];
            if (isNumeric(concatDimSize) && isNumeric(d)) {
                concatDimSize = concatDimSize + d;
            }
            else {
                hasSymbolic = true;
                concatDimSize = `${concatDimSize}+${d}`;
            }
        }
        // Build output shape
        const outputDims = [...firstShape.dims];
        outputDims[dim] = concatDimSize;
        return {
            dims: outputDims,
            description: `Concatenated along dim ${dim}`,
            flags: {
                symbolic: hasSymbolic,
                inferred: true
            },
            provenance: {
                source: 'computed',
                transformation: 'concat',
                description: `Merged ${inputShapes.length} inputs along axis ${dim}`
            }
        };
    }
    /**
     * Validate multiple inputs for concatenation
     */
    validateMultipleInputs(inputShapes, config) {
        if (inputShapes.length < 2) {
            return undefined;
        }
        const dim = config.dim ?? 1;
        const firstShape = inputShapes[0];
        const rank = getRank(firstShape);
        // Check dimension is valid
        if (dim < 0 || dim >= rank) {
            return `Concatenation dimension ${dim} is out of range for ${rank}D tensor`;
        }
        // Check all shapes have same rank
        for (let i = 1; i < inputShapes.length; i++) {
            const inputRank = getRank(inputShapes[i]);
            if (inputRank !== rank) {
                return `Rank mismatch: input 1 is ${rank}D, input ${i + 1} is ${inputRank}D`;
            }
        }
        // Check all non-concat dimensions match
        const conflicts = [];
        for (let d = 0; d < rank; d++) {
            if (d === dim)
                continue;
            const dims = inputShapes.map(s => s.dims[d]);
            const numericDims = dims.filter(isNumeric);
            if (numericDims.length > 1) {
                const unique = new Set(numericDims);
                if (unique.size > 1) {
                    conflicts.push(`dim ${d}: ${dims.join(' vs ')}`);
                }
            }
        }
        if (conflicts.length > 0) {
            return `Dimension mismatch for concat (${conflicts.join('; ')})`;
        }
        return undefined;
    }
    validateIncomingConnection(sourceNodeType, sourceOutputShape, targetConfig) {
        // Concat accepts any input initially
        // Full validation happens when all inputs are connected
        return undefined;
    }
    getDefaultConfig() {
        return {
            dim: 1
        };
    }
}
