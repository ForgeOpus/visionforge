/**
 * Pattern builder functions for declarative shape requirements
 * These provide a clean API for defining node input/output patterns
 */
import { ShapePattern, AxisSpec, AxisRole } from './types';
/**
 * Require exact rank
 * @example exactRank(4) // Must be exactly 4D
 */
export declare function exactRank(n: number): ShapePattern;
/**
 * Require minimum rank
 * @example minRank(2) // Must be at least 2D
 */
export declare function minRank(n: number): ShapePattern;
/**
 * Require rank in range
 * @example rankRange(2, 4) // Must be 2D, 3D, or 4D
 */
export declare function rankRange(min: number, max: number): ShapePattern;
/**
 * Require one of specific ranks
 * @example rankOneOf([2, 4]) // Must be 2D or 4D
 */
export declare function rankOneOf(ranks: number[]): ShapePattern;
/**
 * Create an axis spec with role
 */
export declare function axis(role: AxisRole, options?: Partial<AxisSpec>): AxisSpec;
/**
 * Batch axis (B)
 */
export declare function batchAxis(options?: Partial<AxisSpec>): AxisSpec;
/**
 * Channel axis (C)
 */
export declare function channelAxis(value?: number): AxisSpec;
/**
 * Feature axis (F)
 */
export declare function featureAxis(value?: number): AxisSpec;
/**
 * Time/sequence axis (T)
 */
export declare function timeAxis(options?: Partial<AxisSpec>): AxisSpec;
/**
 * Height axis (H)
 */
export declare function heightAxis(value?: number): AxisSpec;
/**
 * Width axis (W)
 */
export declare function widthAxis(value?: number): AxisSpec;
/**
 * Pattern that ends with a specific axis
 * @example endsWith(featureAxis(256)) // Last dim must be 256 features
 */
export declare function endsWith(axisSpec: AxisSpec): ShapePattern;
/**
 * Pattern that starts with batch dimension
 */
export declare function startsWithBatch(): ShapePattern;
/**
 * Accept any shape (wildcard)
 */
export declare function wildcard(): ShapePattern;
/**
 * Pattern that must be broadcastable to another pattern
 */
export declare function broadcastableTo(targetPattern: ShapePattern): ShapePattern;
/**
 * Standard 2D tensor pattern (B, F)
 * Used by: Linear (strict mode)
 */
export declare function tensor2D(): ShapePattern;
/**
 * Standard 3D tensor pattern (B, T, F)
 * Used by: Sequence models, Attention
 */
export declare function tensor3D(): ShapePattern;
/**
 * Standard 4D image tensor pattern (B, C, H, W)
 * Used by: Conv2D, Pooling
 */
export declare function tensor4D(): ShapePattern;
/**
 * Flexible projection input pattern (*, F_in)
 * Accepts 2D or higher, requires features as last dimension
 * Used by: Linear (with auto_flatten support)
 */
export declare function projectionInput(inFeatures?: number): ShapePattern;
/**
 * Spatial input pattern for convolutions
 * Requires at least 4D with channel as second axis
 */
export declare function spatialInput(inChannels?: number): ShapePattern;
/**
 * Sequence input pattern
 * Accepts 3D with batch, time, features
 */
export declare function sequenceInput(seqLen?: number, features?: number): ShapePattern;
/**
 * Classification output pattern
 * 2D with batch and classes
 */
export declare function classificationOutput(numClasses?: number): ShapePattern;
/**
 * Scalar output pattern (for losses)
 */
export declare function scalarOutput(): ShapePattern;
/**
 * Concatenation merge pattern
 */
export declare function concatMerge(axis?: number): ShapePattern;
/**
 * Addition merge pattern (requires same shape)
 */
export declare function addMerge(): ShapePattern;
/**
 * Stack merge pattern
 */
export declare function stackMerge(axis?: number): ShapePattern;
/**
 * MSE loss input pattern (prediction and target must match)
 */
export declare function mseLossPattern(): {
    prediction: ShapePattern;
    target: ShapePattern;
};
/**
 * Cross-entropy loss input patterns
 */
export declare function crossEntropyPattern(): {
    prediction: ShapePattern;
    target: ShapePattern;
};
/**
 * BCE loss input pattern
 */
export declare function bceLossPattern(): {
    prediction: ShapePattern;
    target: ShapePattern;
};
/**
 * Combine multiple pattern constraints
 */
export declare function combinePatterns(...patterns: ShapePattern[]): ShapePattern;
/**
 * Create a pass-through pattern (output same as input)
 */
export declare function passThrough(): ShapePattern;
//# sourceMappingURL=patterns.d.ts.map