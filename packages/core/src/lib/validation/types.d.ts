/**
 * Core types for the node validation and shape inference system
 */
/**
 * Standard validation result codes with semantic meaning
 */
export declare enum ValidationCode {
    OK = "OK",
    CONFIG_INCOMPLETE = "CONFIG_INCOMPLETE",
    INPUT_SHAPE_PENDING = "INPUT_SHAPE_PENDING",
    PATTERN_MISMATCH_RANK = "PATTERN_MISMATCH_RANK",
    PATTERN_MISMATCH_AXIS = "PATTERN_MISMATCH_AXIS",
    FEATURE_INCOMPATIBLE = "FEATURE_INCOMPATIBLE",
    SYMBOL_UNRESOLVED = "SYMBOL_UNRESOLVED",
    MULTI_INPUT_CONFLICT = "MULTI_INPUT_CONFLICT",
    AUTO_FLATTEN_APPLIED = "AUTO_FLATTEN_APPLIED",
    TARGET_PREDICTION_MISMATCH = "TARGET_PREDICTION_MISMATCH",
    LOSS_INPUT_MISSING = "LOSS_INPUT_MISSING",
    CONNECTION_INVALID = "CONNECTION_INVALID"
}
/**
 * Shape inference status for progressive resolution
 */
export declare enum InferenceStatus {
    /** Shape fully resolved with numeric dimensions */
    RESOLVED = "RESOLVED",
    /** Shape has symbolic dimensions that need resolution */
    SYMBOLIC = "SYMBOLIC",
    /** Waiting for upstream shapes */
    PENDING = "PENDING",
    /** Cannot infer due to configuration issues */
    BLOCKED = "BLOCKED",
    /** Inference failed with error */
    ERROR = "ERROR"
}
/**
 * Node validation state for UI badge display
 */
export declare enum NodeValidationState {
    /** Node not configured */
    UNCONFIGURED = "unconfigured",
    /** Awaiting input connection */
    AWAITING_INPUT = "awaiting_input",
    /** Negotiating symbols or auto-flatten applied */
    NEGOTIATING = "negotiating",
    /** Pattern mismatch or feature incompatible */
    ERROR = "error",
    /** Fully resolved and valid */
    VALID = "valid"
}
/**
 * Semantic roles for tensor axes
 */
export type AxisRole = 'batch' | 'time' | 'channels' | 'height' | 'width' | 'features' | 'vocab' | 'embedding' | 'classes' | 'heads' | 'any';
/**
 * Common symbolic dimension names
 */
export declare const SYMBOLS: {
    readonly BATCH: "B";
    readonly TIME: "T";
    readonly CHANNELS: "C";
    readonly HEIGHT: "H";
    readonly WIDTH: "W";
    readonly FEATURES: "F";
    readonly VOCAB: "V";
    readonly EMBEDDING: "D";
    readonly CLASSES: "K";
    readonly HEADS: "N";
};
/**
 * Specification for a single axis in a shape pattern
 */
export interface AxisSpec {
    /** Semantic role of this axis */
    role?: AxisRole;
    /** Symbolic name (e.g., 'B', 'T', 'F') */
    symbol?: string;
    /** Fixed numeric value */
    numeric?: number;
    /** Whether this axis can be any size */
    flexible?: boolean;
    /** Minimum allowed value */
    min?: number;
    /** Maximum allowed value */
    max?: number;
}
/**
 * A dimension value that can be numeric or symbolic
 */
export type DimensionValue = number | string;
/**
 * Rank requirement specification
 */
export interface RankSpec {
    /** Exact rank required */
    exact?: number;
    /** Minimum rank required */
    min?: number;
    /** Maximum rank allowed */
    max?: number;
    /** List of allowed ranks */
    oneOf?: number[];
}
/**
 * Merge operation specification for multi-input nodes
 */
export interface MergeSpec {
    /** Type of merge operation */
    mode: 'concat' | 'add' | 'stack' | 'multiply';
    /** Axis along which to merge (for concat/stack) */
    axis?: number;
    /** Constraints for axis alignment */
    axisConstraints?: {
        /** Axes that must match exactly */
        mustMatch: number[];
        /** Axes that can differ (for concat) */
        canDiffer?: number[];
    };
}
/**
 * Pattern specification for shape validation
 * Used to define what shapes a node accepts or produces
 */
export interface ShapePattern {
    /** Rank requirements */
    rank?: number | RankSpec;
    /** Ordered list of axis specifications */
    axes?: AxisSpec[];
    /** Constraint on the last axis (e.g., must be features) */
    endsWith?: AxisSpec;
    /** Constraint on the first axis (usually batch) */
    startsWith?: AxisSpec;
    /** Accept any shape */
    wildcard?: boolean;
    /** Must be broadcastable to this pattern */
    broadcastableTo?: ShapePattern;
    /** Merge rule for multi-input patterns */
    mergeRule?: MergeSpec;
    /** Preserve leading N dimensions unchanged */
    preserveLeading?: number;
    /** Human-readable description of this pattern */
    description?: string;
}
/**
 * Flags that modify shape behavior
 */
export interface ShapeFlags {
    /** Automatically flatten before processing */
    autoFlatten?: boolean;
    /** Shape was inferred (not explicitly set) */
    inferred?: boolean;
    /** Shape contains symbolic dimensions */
    symbolic?: boolean;
}
/**
 * Provenance information for how a shape was derived
 */
export interface ShapeProvenance {
    /** Source of the shape (e.g., 'user', 'inferred', 'computed') */
    source: 'user' | 'inferred' | 'computed' | 'default';
    /** Node ID that produced this shape */
    fromNodeId?: string;
    /** Transformation applied (e.g., 'flatten', 'conv2d') */
    transformation?: string;
    /** Description of how the shape was derived */
    description?: string;
}
/**
 * Extended tensor shape with validation metadata
 */
export interface ExtendedTensorShape {
    /** Dimension values (numeric or symbolic) */
    dims: DimensionValue[];
    /** Human-readable description */
    description?: string;
    /** Shape pattern this conforms to */
    pattern?: ShapePattern;
    /** Behavior flags */
    flags?: ShapeFlags;
    /** How this shape was derived */
    provenance?: ShapeProvenance;
}
/**
 * Result of a connection validation
 */
export interface ValidationResult {
    /** Whether the connection is valid */
    ok: boolean;
    /** Validation result code */
    code: ValidationCode;
    /** Human-friendly message */
    message: string;
    /** Normalized shape after any transformations */
    normalizedShape?: ExtendedTensorShape;
    /** Suggested action to fix the issue */
    actionHint?: string;
    /** Additional context for debugging */
    details?: Record<string, unknown>;
}
/**
 * Result of shape inference
 */
export interface InferenceResult {
    /** Inferred output shape */
    shape?: ExtendedTensorShape;
    /** Inference status */
    status: InferenceStatus;
    /** Validation code if there's an issue */
    code?: ValidationCode;
    /** Human-friendly message */
    message?: string;
    /** Suggested action */
    actionHint?: string;
}
/**
 * Conflict information for merge operations
 */
export interface MergeConflict {
    /** Axis index where conflict occurs */
    axis: number;
    /** Values that conflict */
    values: DimensionValue[];
    /** Source node IDs */
    sourceNodeIds: string[];
    /** Description of the conflict */
    description: string;
}
/**
 * Result of merge negotiation
 */
export interface MergeResult {
    /** Negotiated output shape */
    shape?: ExtendedTensorShape;
    /** List of conflicts if merge failed */
    conflicts?: MergeConflict[];
    /** Whether merge is valid */
    ok: boolean;
    /** Validation code */
    code: ValidationCode;
    /** Message */
    message: string;
}
/**
 * Complete shape status for a node
 */
export interface NodeShapeStatus {
    /** Overall validation state */
    state: NodeValidationState;
    /** Input shape(s) received */
    inputShapes: ExtendedTensorShape[];
    /** Computed output shape */
    outputShape?: ExtendedTensorShape;
    /** Validation result for inputs */
    inputValidation?: ValidationResult;
    /** Inference result for output */
    outputInference?: InferenceResult;
    /** Timestamp of last validation */
    timestamp: number;
}
/**
 * Context information for validation operations
 */
export interface ValidationContext {
    /** ID of the source node */
    sourceNodeId: string;
    /** ID of the target node */
    targetNodeId: string;
    /** Source port ID */
    sourcePortId: string;
    /** Target port ID */
    targetPortId: string;
    /** Target node configuration */
    targetConfig: Record<string, unknown>;
    /** Symbol mappings for resolution */
    symbolMappings?: Record<string, number>;
}
/**
 * Context for shape inference
 */
export interface InferenceContext {
    /** Node ID */
    nodeId: string;
    /** Node type */
    nodeType: string;
    /** Node configuration */
    config: Record<string, unknown>;
    /** Symbol mappings */
    symbolMappings?: Record<string, number>;
    /** Whether to apply auto-flatten */
    autoFlatten?: boolean;
}
//# sourceMappingURL=types.d.ts.map