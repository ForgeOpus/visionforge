/**
 * Core types for the node validation and shape inference system
 */
// =============================================================================
// Validation Codes & Status
// =============================================================================
/**
 * Standard validation result codes with semantic meaning
 */
export var ValidationCode;
(function (ValidationCode) {
    ValidationCode["OK"] = "OK";
    ValidationCode["CONFIG_INCOMPLETE"] = "CONFIG_INCOMPLETE";
    ValidationCode["INPUT_SHAPE_PENDING"] = "INPUT_SHAPE_PENDING";
    ValidationCode["PATTERN_MISMATCH_RANK"] = "PATTERN_MISMATCH_RANK";
    ValidationCode["PATTERN_MISMATCH_AXIS"] = "PATTERN_MISMATCH_AXIS";
    ValidationCode["FEATURE_INCOMPATIBLE"] = "FEATURE_INCOMPATIBLE";
    ValidationCode["SYMBOL_UNRESOLVED"] = "SYMBOL_UNRESOLVED";
    ValidationCode["MULTI_INPUT_CONFLICT"] = "MULTI_INPUT_CONFLICT";
    ValidationCode["AUTO_FLATTEN_APPLIED"] = "AUTO_FLATTEN_APPLIED";
    ValidationCode["TARGET_PREDICTION_MISMATCH"] = "TARGET_PREDICTION_MISMATCH";
    ValidationCode["LOSS_INPUT_MISSING"] = "LOSS_INPUT_MISSING";
    ValidationCode["CONNECTION_INVALID"] = "CONNECTION_INVALID";
})(ValidationCode || (ValidationCode = {}));
/**
 * Shape inference status for progressive resolution
 */
export var InferenceStatus;
(function (InferenceStatus) {
    /** Shape fully resolved with numeric dimensions */
    InferenceStatus["RESOLVED"] = "RESOLVED";
    /** Shape has symbolic dimensions that need resolution */
    InferenceStatus["SYMBOLIC"] = "SYMBOLIC";
    /** Waiting for upstream shapes */
    InferenceStatus["PENDING"] = "PENDING";
    /** Cannot infer due to configuration issues */
    InferenceStatus["BLOCKED"] = "BLOCKED";
    /** Inference failed with error */
    InferenceStatus["ERROR"] = "ERROR";
})(InferenceStatus || (InferenceStatus = {}));
/**
 * Node validation state for UI badge display
 */
export var NodeValidationState;
(function (NodeValidationState) {
    /** Node not configured */
    NodeValidationState["UNCONFIGURED"] = "unconfigured";
    /** Awaiting input connection */
    NodeValidationState["AWAITING_INPUT"] = "awaiting_input";
    /** Negotiating symbols or auto-flatten applied */
    NodeValidationState["NEGOTIATING"] = "negotiating";
    /** Pattern mismatch or feature incompatible */
    NodeValidationState["ERROR"] = "error";
    /** Fully resolved and valid */
    NodeValidationState["VALID"] = "valid";
})(NodeValidationState || (NodeValidationState = {}));
/**
 * Common symbolic dimension names
 */
export const SYMBOLS = {
    BATCH: 'B',
    TIME: 'T',
    CHANNELS: 'C',
    HEIGHT: 'H',
    WIDTH: 'W',
    FEATURES: 'F',
    VOCAB: 'V',
    EMBEDDING: 'D',
    CLASSES: 'K',
    HEADS: 'N',
};
