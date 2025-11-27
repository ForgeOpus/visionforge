/**
 * Pattern matching functions for shape validation
 */
import { ValidationCode, } from './types';
// =============================================================================
// Utility Functions
// =============================================================================
/**
 * Check if a dimension is numeric
 */
export function isNumeric(dim) {
    return typeof dim === 'number';
}
/**
 * Check if a dimension is symbolic
 */
export function isSymbolic(dim) {
    return typeof dim === 'string';
}
/**
 * Get the rank (number of dimensions) of a shape
 */
export function getRank(shape) {
    return shape.dims.length;
}
/**
 * Check if all dimensions are numeric
 */
export function isFullyNumeric(shape) {
    return shape.dims.every(isNumeric);
}
/**
 * Check if shape has any symbolic dimensions
 */
export function hasSymbolicDims(shape) {
    return shape.dims.some(isSymbolic);
}
/**
 * Get the last dimension
 */
export function getLastDim(shape) {
    return shape.dims[shape.dims.length - 1];
}
/**
 * Get the first dimension (usually batch)
 */
export function getFirstDim(shape) {
    return shape.dims[0];
}
// =============================================================================
// Rank Matching
// =============================================================================
/**
 * Check if shape rank matches the rank specification
 */
export function matchRank(shape, rankSpec) {
    const rank = getRank(shape);
    if (typeof rankSpec === 'number') {
        if (rank !== rankSpec) {
            return {
                matches: false,
                code: ValidationCode.PATTERN_MISMATCH_RANK,
                message: `Expected rank ${rankSpec}, got ${rank}`,
            };
        }
        return { matches: true, code: ValidationCode.OK, message: 'Rank matches' };
    }
    // RankSpec object
    if (rankSpec.exact !== undefined && rank !== rankSpec.exact) {
        return {
            matches: false,
            code: ValidationCode.PATTERN_MISMATCH_RANK,
            message: `Expected exactly ${rankSpec.exact}D, got ${rank}D`,
        };
    }
    if (rankSpec.min !== undefined && rank < rankSpec.min) {
        return {
            matches: false,
            code: ValidationCode.PATTERN_MISMATCH_RANK,
            message: `Expected at least ${rankSpec.min}D, got ${rank}D`,
        };
    }
    if (rankSpec.max !== undefined && rank > rankSpec.max) {
        return {
            matches: false,
            code: ValidationCode.PATTERN_MISMATCH_RANK,
            message: `Expected at most ${rankSpec.max}D, got ${rank}D`,
        };
    }
    if (rankSpec.oneOf !== undefined && !rankSpec.oneOf.includes(rank)) {
        return {
            matches: false,
            code: ValidationCode.PATTERN_MISMATCH_RANK,
            message: `Expected ${rankSpec.oneOf.join('D or ')}D, got ${rank}D`,
        };
    }
    return { matches: true, code: ValidationCode.OK, message: 'Rank matches' };
}
// =============================================================================
// Axis Matching
// =============================================================================
/**
 * Check if a dimension matches an axis specification
 */
export function matchAxis(dim, axisSpec, axisIndex) {
    // If numeric value is specified, must match exactly
    if (axisSpec.numeric !== undefined) {
        if (isNumeric(dim) && dim !== axisSpec.numeric) {
            return {
                matches: false,
                code: ValidationCode.FEATURE_INCOMPATIBLE,
                message: `Axis ${axisIndex}: expected ${axisSpec.numeric}, got ${dim}`,
            };
        }
        // Symbolic dimensions are allowed if they might resolve to the correct value
    }
    // Check min/max constraints
    if (isNumeric(dim)) {
        if (axisSpec.min !== undefined && dim < axisSpec.min) {
            return {
                matches: false,
                code: ValidationCode.PATTERN_MISMATCH_AXIS,
                message: `Axis ${axisIndex}: ${dim} is less than minimum ${axisSpec.min}`,
            };
        }
        if (axisSpec.max !== undefined && dim > axisSpec.max) {
            return {
                matches: false,
                code: ValidationCode.PATTERN_MISMATCH_AXIS,
                message: `Axis ${axisIndex}: ${dim} is greater than maximum ${axisSpec.max}`,
            };
        }
    }
    // If flexible, any value is acceptable
    if (axisSpec.flexible) {
        return { matches: true, code: ValidationCode.OK, message: 'Axis matches (flexible)' };
    }
    return { matches: true, code: ValidationCode.OK, message: 'Axis matches' };
}
// =============================================================================
// Pattern Matching
// =============================================================================
/**
 * Match a shape against a pattern
 */
export function matchPattern(shape, pattern) {
    // Wildcard accepts everything
    if (pattern.wildcard) {
        return { matches: true, code: ValidationCode.OK, message: 'Wildcard pattern matches' };
    }
    // Check rank first
    if (pattern.rank !== undefined) {
        const rankResult = matchRank(shape, pattern.rank);
        if (!rankResult.matches) {
            return rankResult;
        }
    }
    // Check startsWith constraint
    if (pattern.startsWith !== undefined && shape.dims.length > 0) {
        const firstDim = shape.dims[0];
        const result = matchAxis(firstDim, pattern.startsWith, 0);
        if (!result.matches) {
            return result;
        }
    }
    // Check endsWith constraint
    if (pattern.endsWith !== undefined && shape.dims.length > 0) {
        const lastDim = shape.dims[shape.dims.length - 1];
        const result = matchAxis(lastDim, pattern.endsWith, shape.dims.length - 1);
        if (!result.matches) {
            return result;
        }
    }
    // Check axes if specified
    if (pattern.axes !== undefined) {
        if (shape.dims.length !== pattern.axes.length) {
            return {
                matches: false,
                code: ValidationCode.PATTERN_MISMATCH_RANK,
                message: `Expected ${pattern.axes.length} axes, got ${shape.dims.length}`,
            };
        }
        for (let i = 0; i < pattern.axes.length; i++) {
            const result = matchAxis(shape.dims[i], pattern.axes[i], i);
            if (!result.matches) {
                return result;
            }
        }
    }
    return { matches: true, code: ValidationCode.OK, message: 'Pattern matches' };
}
// =============================================================================
// Feature Compatibility
// =============================================================================
/**
 * Check if two feature dimensions are compatible
 */
export function areFeaturesCompatible(sourceDim, targetDim) {
    // Both numeric - must match
    if (isNumeric(sourceDim) && isNumeric(targetDim)) {
        if (sourceDim !== targetDim) {
            return {
                matches: false,
                code: ValidationCode.FEATURE_INCOMPATIBLE,
                message: `Feature dimension mismatch: got ${sourceDim}, expected ${targetDim}`,
            };
        }
        return { matches: true, code: ValidationCode.OK, message: 'Features compatible' };
    }
    // If target is symbolic, source must be symbolic with same symbol or numeric
    if (isSymbolic(targetDim)) {
        if (isSymbolic(sourceDim) && sourceDim !== targetDim) {
            return {
                matches: false,
                code: ValidationCode.SYMBOL_UNRESOLVED,
                message: `Symbol mismatch: got ${sourceDim}, expected ${targetDim}`,
            };
        }
        // Numeric source with symbolic target - will need to bind symbol later
        return { matches: true, code: ValidationCode.OK, message: 'Features compatible (symbolic)' };
    }
    // Numeric target with symbolic source - may resolve
    if (isSymbolic(sourceDim)) {
        return {
            matches: true,
            code: ValidationCode.SYMBOL_UNRESOLVED,
            message: `Symbol ${sourceDim} needs to resolve to ${targetDim}`,
        };
    }
    return { matches: true, code: ValidationCode.OK, message: 'Features compatible' };
}
// =============================================================================
// Shape Transformations
// =============================================================================
/**
 * Flatten a shape from start_dim to end_dim
 */
export function flattenShape(shape, startDim = 1, endDim = -1) {
    const rank = getRank(shape);
    // Normalize negative indices
    const normalizedStart = startDim < 0 ? rank + startDim : startDim;
    const normalizedEnd = endDim < 0 ? rank + endDim : endDim;
    if (normalizedStart < 0 || normalizedStart >= rank) {
        return null;
    }
    if (normalizedEnd < normalizedStart || normalizedEnd >= rank) {
        return null;
    }
    // Compute flattened dimension
    const dimsToFlatten = shape.dims.slice(normalizedStart, normalizedEnd + 1);
    // If all numeric, compute product
    if (dimsToFlatten.every(isNumeric)) {
        const product = dimsToFlatten.reduce((a, b) => a * b, 1);
        return {
            dims: [
                ...shape.dims.slice(0, normalizedStart),
                product,
                ...shape.dims.slice(normalizedEnd + 1),
            ],
            description: `Flattened from ${shape.dims.join('×')}`,
            flags: { inferred: true },
            provenance: {
                source: 'computed',
                transformation: 'flatten',
            },
        };
    }
    // If symbolic, create symbolic product
    const flatDim = dimsToFlatten.map(d => String(d)).join('*');
    return {
        dims: [
            ...shape.dims.slice(0, normalizedStart),
            flatDim,
            ...shape.dims.slice(normalizedEnd + 1),
        ],
        description: `Flattened with symbolic dimensions`,
        flags: { symbolic: true, inferred: true },
        provenance: {
            source: 'computed',
            transformation: 'flatten',
        },
    };
}
/**
 * Apply auto-flatten for projection layers
 * Flattens all dimensions except batch into features
 */
export function autoFlattenForProjection(shape) {
    if (getRank(shape) <= 2) {
        return shape;
    }
    return flattenShape(shape, 1, -1) || shape;
}
// =============================================================================
// Merge Operations
// =============================================================================
/**
 * Check if shapes can be merged via concatenation
 */
export function canConcatenate(shapes, axis = 1) {
    if (shapes.length < 2) {
        return { matches: true, code: ValidationCode.OK, message: 'Single input' };
    }
    const refShape = shapes[0];
    const refRank = getRank(refShape);
    for (let i = 1; i < shapes.length; i++) {
        const shape = shapes[i];
        if (getRank(shape) !== refRank) {
            return {
                matches: false,
                code: ValidationCode.MULTI_INPUT_CONFLICT,
                message: `Rank mismatch for concatenation: ${refRank}D vs ${getRank(shape)}D`,
            };
        }
        // Check all dimensions except concat axis
        for (let d = 0; d < refRank; d++) {
            if (d === axis)
                continue;
            const refDim = refShape.dims[d];
            const curDim = shape.dims[d];
            if (isNumeric(refDim) && isNumeric(curDim) && refDim !== curDim) {
                return {
                    matches: false,
                    code: ValidationCode.MULTI_INPUT_CONFLICT,
                    message: `Dimension ${d} mismatch: ${refDim} vs ${curDim}`,
                };
            }
        }
    }
    return { matches: true, code: ValidationCode.OK, message: 'Can concatenate' };
}
/**
 * Check if shapes can be merged via element-wise addition
 */
export function canAdd(shapes) {
    if (shapes.length < 2) {
        return { matches: true, code: ValidationCode.OK, message: 'Single input' };
    }
    const refShape = shapes[0];
    const refRank = getRank(refShape);
    for (let i = 1; i < shapes.length; i++) {
        const shape = shapes[i];
        if (getRank(shape) !== refRank) {
            return {
                matches: false,
                code: ValidationCode.MULTI_INPUT_CONFLICT,
                message: `Rank mismatch for addition: ${refRank}D vs ${getRank(shape)}D`,
            };
        }
        // All dimensions must match
        for (let d = 0; d < refRank; d++) {
            const refDim = refShape.dims[d];
            const curDim = shape.dims[d];
            if (isNumeric(refDim) && isNumeric(curDim) && refDim !== curDim) {
                return {
                    matches: false,
                    code: ValidationCode.MULTI_INPUT_CONFLICT,
                    message: `Dimension ${d} mismatch for addition: ${refDim} vs ${curDim}`,
                };
            }
        }
    }
    return { matches: true, code: ValidationCode.OK, message: 'Can add' };
}
/**
 * Compute output shape after concatenation
 */
export function computeConcatShape(shapes, axis = 1) {
    if (shapes.length === 0)
        return null;
    if (shapes.length === 1)
        return shapes[0];
    const canConcat = canConcatenate(shapes, axis);
    if (!canConcat.matches)
        return null;
    const result = [...shapes[0].dims];
    // Sum along concat axis
    let axisSum = 0;
    let hasSymbolic = false;
    for (const shape of shapes) {
        const dim = shape.dims[axis];
        if (isNumeric(axisSum) && isNumeric(dim)) {
            axisSum = axisSum + dim;
        }
        else {
            hasSymbolic = true;
            axisSum = `${axisSum}+${dim}`;
        }
    }
    result[axis] = axisSum;
    return {
        dims: result,
        description: `Concatenated along axis ${axis}`,
        flags: { symbolic: hasSymbolic, inferred: true },
        provenance: {
            source: 'computed',
            transformation: 'concat',
        },
    };
}
