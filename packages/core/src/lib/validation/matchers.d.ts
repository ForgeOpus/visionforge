/**
 * Pattern matching functions for shape validation
 */
import type { TensorShape } from '../types';
import { ShapePattern, AxisSpec, RankSpec, ValidationCode, DimensionValue, ExtendedTensorShape } from './types';
export interface MatchResult {
    matches: boolean;
    code: ValidationCode;
    message: string;
    normalizedShape?: ExtendedTensorShape;
}
/**
 * Check if a dimension is numeric
 */
export declare function isNumeric(dim: DimensionValue): dim is number;
/**
 * Check if a dimension is symbolic
 */
export declare function isSymbolic(dim: DimensionValue): dim is string;
/**
 * Get the rank (number of dimensions) of a shape
 */
export declare function getRank(shape: TensorShape): number;
/**
 * Check if all dimensions are numeric
 */
export declare function isFullyNumeric(shape: TensorShape): boolean;
/**
 * Check if shape has any symbolic dimensions
 */
export declare function hasSymbolicDims(shape: TensorShape): boolean;
/**
 * Get the last dimension
 */
export declare function getLastDim(shape: TensorShape): DimensionValue | undefined;
/**
 * Get the first dimension (usually batch)
 */
export declare function getFirstDim(shape: TensorShape): DimensionValue | undefined;
/**
 * Check if shape rank matches the rank specification
 */
export declare function matchRank(shape: TensorShape, rankSpec: number | RankSpec): MatchResult;
/**
 * Check if a dimension matches an axis specification
 */
export declare function matchAxis(dim: DimensionValue, axisSpec: AxisSpec, axisIndex: number): MatchResult;
/**
 * Match a shape against a pattern
 */
export declare function matchPattern(shape: TensorShape, pattern: ShapePattern): MatchResult;
/**
 * Check if two feature dimensions are compatible
 */
export declare function areFeaturesCompatible(sourceDim: DimensionValue, targetDim: DimensionValue): MatchResult;
/**
 * Flatten a shape from start_dim to end_dim
 */
export declare function flattenShape(shape: TensorShape, startDim?: number, endDim?: number): TensorShape | null;
/**
 * Apply auto-flatten for projection layers
 * Flattens all dimensions except batch into features
 */
export declare function autoFlattenForProjection(shape: TensorShape): TensorShape;
/**
 * Check if shapes can be merged via concatenation
 */
export declare function canConcatenate(shapes: TensorShape[], axis?: number): MatchResult;
/**
 * Check if shapes can be merged via element-wise addition
 */
export declare function canAdd(shapes: TensorShape[]): MatchResult;
/**
 * Compute output shape after concatenation
 */
export declare function computeConcatShape(shapes: TensorShape[], axis?: number): TensorShape | null;
//# sourceMappingURL=matchers.d.ts.map