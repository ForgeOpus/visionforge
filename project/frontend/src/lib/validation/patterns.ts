/**
 * Pattern builder functions for declarative shape requirements
 * These provide a clean API for defining node input/output patterns
 */

import {
  ShapePattern,
  AxisSpec,
  AxisRole,
  MergeSpec,
  RankSpec,
  SYMBOLS,
} from './types';

// =============================================================================
// Rank Pattern Builders
// =============================================================================

/**
 * Require exact rank
 * @example exactRank(4) // Must be exactly 4D
 */
export function exactRank(n: number): ShapePattern {
  return {
    rank: { exact: n },
    description: `Exactly ${n}D tensor`,
  };
}

/**
 * Require minimum rank
 * @example minRank(2) // Must be at least 2D
 */
export function minRank(n: number): ShapePattern {
  return {
    rank: { min: n },
    description: `At least ${n}D tensor`,
  };
}

/**
 * Require rank in range
 * @example rankRange(2, 4) // Must be 2D, 3D, or 4D
 */
export function rankRange(min: number, max: number): ShapePattern {
  return {
    rank: { min, max },
    description: `${min}D to ${max}D tensor`,
  };
}

/**
 * Require one of specific ranks
 * @example rankOneOf([2, 4]) // Must be 2D or 4D
 */
export function rankOneOf(ranks: number[]): ShapePattern {
  return {
    rank: { oneOf: ranks },
    description: `${ranks.join('D or ')}D tensor`,
  };
}

// =============================================================================
// Axis Pattern Builders
// =============================================================================

/**
 * Create an axis spec with role
 */
export function axis(role: AxisRole, options?: Partial<AxisSpec>): AxisSpec {
  return {
    role,
    ...options,
  };
}

/**
 * Batch axis (B)
 */
export function batchAxis(options?: Partial<AxisSpec>): AxisSpec {
  return {
    role: 'batch',
    symbol: SYMBOLS.BATCH,
    flexible: true,
    ...options,
  };
}

/**
 * Channel axis (C)
 */
export function channelAxis(value?: number): AxisSpec {
  return {
    role: 'channels',
    symbol: SYMBOLS.CHANNELS,
    numeric: value,
  };
}

/**
 * Feature axis (F)
 */
export function featureAxis(value?: number): AxisSpec {
  return {
    role: 'features',
    symbol: SYMBOLS.FEATURES,
    numeric: value,
  };
}

/**
 * Time/sequence axis (T)
 */
export function timeAxis(options?: Partial<AxisSpec>): AxisSpec {
  return {
    role: 'time',
    symbol: SYMBOLS.TIME,
    flexible: true,
    ...options,
  };
}

/**
 * Height axis (H)
 */
export function heightAxis(value?: number): AxisSpec {
  return {
    role: 'height',
    symbol: SYMBOLS.HEIGHT,
    numeric: value,
  };
}

/**
 * Width axis (W)
 */
export function widthAxis(value?: number): AxisSpec {
  return {
    role: 'width',
    symbol: SYMBOLS.WIDTH,
    numeric: value,
  };
}

// =============================================================================
// Composite Pattern Builders
// =============================================================================

/**
 * Pattern that ends with a specific axis
 * @example endsWith(featureAxis(256)) // Last dim must be 256 features
 */
export function endsWith(axisSpec: AxisSpec): ShapePattern {
  return {
    endsWith: axisSpec,
    description: `Ends with ${axisSpec.role || 'specified'} axis`,
  };
}

/**
 * Pattern that starts with batch dimension
 */
export function startsWithBatch(): ShapePattern {
  return {
    startsWith: batchAxis(),
    description: 'Starts with batch dimension',
  };
}

/**
 * Accept any shape (wildcard)
 */
export function wildcard(): ShapePattern {
  return {
    wildcard: true,
    description: 'Any shape accepted',
  };
}

/**
 * Pattern that must be broadcastable to another pattern
 */
export function broadcastableTo(targetPattern: ShapePattern): ShapePattern {
  return {
    broadcastableTo: targetPattern,
    description: `Broadcastable to ${targetPattern.description || 'target'}`,
  };
}

// =============================================================================
// Common Node Patterns
// =============================================================================

/**
 * Standard 2D tensor pattern (B, F)
 * Used by: Linear (strict mode)
 */
export function tensor2D(): ShapePattern {
  return {
    rank: { exact: 2 },
    axes: [batchAxis(), featureAxis()],
    description: '2D tensor (batch, features)',
  };
}

/**
 * Standard 3D tensor pattern (B, T, F)
 * Used by: Sequence models, Attention
 */
export function tensor3D(): ShapePattern {
  return {
    rank: { exact: 3 },
    axes: [batchAxis(), timeAxis(), featureAxis()],
    description: '3D tensor (batch, time, features)',
  };
}

/**
 * Standard 4D image tensor pattern (B, C, H, W)
 * Used by: Conv2D, Pooling
 */
export function tensor4D(): ShapePattern {
  return {
    rank: { exact: 4 },
    axes: [batchAxis(), channelAxis(), heightAxis(), widthAxis()],
    description: '4D tensor (batch, channels, height, width)',
  };
}

/**
 * Flexible projection input pattern (*, F_in)
 * Accepts 2D or higher, requires features as last dimension
 * Used by: Linear (with auto_flatten support)
 */
export function projectionInput(inFeatures?: number): ShapePattern {
  return {
    rank: { min: 2 },
    endsWith: featureAxis(inFeatures),
    preserveLeading: 1, // Preserve batch
    description: inFeatures
      ? `(*, ${inFeatures}) - projection input`
      : '(*, features) - projection input',
  };
}

/**
 * Spatial input pattern for convolutions
 * Requires at least 4D with channel as second axis
 */
export function spatialInput(inChannels?: number): ShapePattern {
  return {
    rank: { min: 4 },
    axes: [
      batchAxis(),
      channelAxis(inChannels),
      heightAxis(),
      widthAxis(),
    ],
    description: inChannels
      ? `(B, ${inChannels}, H, W) - spatial input`
      : '(B, C, H, W) - spatial input',
  };
}

/**
 * Sequence input pattern
 * Accepts 3D with batch, time, features
 */
export function sequenceInput(seqLen?: number, features?: number): ShapePattern {
  return {
    rank: { exact: 3 },
    axes: [
      batchAxis(),
      timeAxis(seqLen ? { numeric: seqLen } : undefined),
      featureAxis(features),
    ],
    description: '(B, T, F) - sequence input',
  };
}

/**
 * Classification output pattern
 * 2D with batch and classes
 */
export function classificationOutput(numClasses?: number): ShapePattern {
  return {
    rank: { exact: 2 },
    axes: [
      batchAxis(),
      { role: 'classes', symbol: SYMBOLS.CLASSES, numeric: numClasses },
    ],
    description: numClasses
      ? `(B, ${numClasses}) - classification output`
      : '(B, K) - classification output',
  };
}

/**
 * Scalar output pattern (for losses)
 */
export function scalarOutput(): ShapePattern {
  return {
    rank: { exact: 0 },
    description: 'Scalar value',
  };
}

// =============================================================================
// Merge Patterns
// =============================================================================

/**
 * Concatenation merge pattern
 */
export function concatMerge(axis: number = 1): ShapePattern {
  return {
    wildcard: true, // Accept any rank initially
    mergeRule: {
      mode: 'concat',
      axis,
      axisConstraints: {
        mustMatch: [], // Will be filled based on input ranks
        canDiffer: [axis],
      },
    },
    description: `Concatenate along axis ${axis}`,
  };
}

/**
 * Addition merge pattern (requires same shape)
 */
export function addMerge(): ShapePattern {
  return {
    wildcard: true,
    mergeRule: {
      mode: 'add',
      axisConstraints: {
        mustMatch: [], // All axes must match
      },
    },
    description: 'Element-wise addition (same shape)',
  };
}

/**
 * Stack merge pattern
 */
export function stackMerge(axis: number = 0): ShapePattern {
  return {
    wildcard: true,
    mergeRule: {
      mode: 'stack',
      axis,
      axisConstraints: {
        mustMatch: [], // All existing axes must match
      },
    },
    description: `Stack along new axis ${axis}`,
  };
}

// =============================================================================
// Loss Patterns
// =============================================================================

/**
 * MSE loss input pattern (prediction and target must match)
 */
export function mseLossPattern(): { prediction: ShapePattern; target: ShapePattern } {
  return {
    prediction: wildcard(),
    target: wildcard(), // Must match prediction exactly
  };
}

/**
 * Cross-entropy loss input patterns
 */
export function crossEntropyPattern(): { prediction: ShapePattern; target: ShapePattern } {
  return {
    prediction: {
      rank: { oneOf: [2, 3] }, // (B, K) or (B, T, K)
      endsWith: { role: 'classes', symbol: SYMBOLS.CLASSES },
      description: 'Logits (B, K) or (B, T, K)',
    },
    target: {
      rank: { oneOf: [1, 2] }, // (B,) or (B, T) indices
      description: 'Class indices (B,) or (B, T)',
    },
  };
}

/**
 * BCE loss input pattern
 */
export function bceLossPattern(): { prediction: ShapePattern; target: ShapePattern } {
  return {
    prediction: wildcard(),
    target: wildcard(), // Must match or be broadcastable
  };
}

// =============================================================================
// Pattern Combination Utilities
// =============================================================================

/**
 * Combine multiple pattern constraints
 */
export function combinePatterns(...patterns: ShapePattern[]): ShapePattern {
  const combined: ShapePattern = {};

  for (const pattern of patterns) {
    if (pattern.rank) combined.rank = pattern.rank;
    if (pattern.axes) combined.axes = pattern.axes;
    if (pattern.endsWith) combined.endsWith = pattern.endsWith;
    if (pattern.startsWith) combined.startsWith = pattern.startsWith;
    if (pattern.wildcard) combined.wildcard = pattern.wildcard;
    if (pattern.broadcastableTo) combined.broadcastableTo = pattern.broadcastableTo;
    if (pattern.mergeRule) combined.mergeRule = pattern.mergeRule;
    if (pattern.preserveLeading !== undefined) combined.preserveLeading = pattern.preserveLeading;
    if (pattern.description) {
      combined.description = combined.description
        ? `${combined.description}; ${pattern.description}`
        : pattern.description;
    }
  }

  return combined;
}

/**
 * Create a pass-through pattern (output same as input)
 */
export function passThrough(): ShapePattern {
  return {
    wildcard: true,
    description: 'Pass-through (output same as input)',
  };
}
