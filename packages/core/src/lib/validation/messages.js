/**
 * User-friendly message mapping for validation codes
 */
import { ValidationCode, NodeValidationState } from './types';
/**
 * Get user-friendly message for a validation code
 */
export function getValidationMessage(code, context) {
    switch (code) {
        case ValidationCode.OK:
            return {
                title: 'Valid',
                description: 'Configuration and connections are valid',
                actionHint: '',
            };
        case ValidationCode.CONFIG_INCOMPLETE:
            return {
                title: 'Configuration Incomplete',
                description: context?.field
                    ? `${context.field} is not set`
                    : 'Required configuration fields are missing',
                actionHint: 'Enter required values in the configuration panel',
            };
        case ValidationCode.INPUT_SHAPE_PENDING:
            return {
                title: 'Waiting for Input',
                description: 'Upstream shape not yet available',
                actionHint: 'Connect a source node or configure upstream',
            };
        case ValidationCode.PATTERN_MISMATCH_RANK:
            return {
                title: 'Rank Mismatch',
                description: context?.expected
                    ? `Expected ${context.expected}D tensor, got ${context.actual}D`
                    : 'Tensor rank does not match requirements',
                actionHint: 'Insert Reshape or Flatten to adjust tensor dimensions',
            };
        case ValidationCode.PATTERN_MISMATCH_AXIS:
            return {
                title: 'Axis Mismatch',
                description: context?.axis
                    ? `Axis ${context.axis} does not match requirements`
                    : 'Tensor axis does not match pattern',
                actionHint: 'Check tensor dimensions at specified axis',
            };
        case ValidationCode.FEATURE_INCOMPATIBLE:
            return {
                title: 'Feature Dimension Mismatch',
                description: context?.expected
                    ? `Expected ${context.expected} features, got ${context.actual}`
                    : 'Feature dimensions do not match',
                actionHint: 'Align in_features with upstream output or add projection layer',
            };
        case ValidationCode.SYMBOL_UNRESOLVED:
            return {
                title: 'Unresolved Symbol',
                description: context?.symbol
                    ? `Symbol ${context.symbol} needs resolution`
                    : 'Symbolic dimension cannot be resolved',
                actionHint: 'Unify symbolic dimensions via configuration or upstream settings',
            };
        case ValidationCode.MULTI_INPUT_CONFLICT:
            return {
                title: 'Input Conflict',
                description: context?.axis
                    ? `Inputs differ at axis ${context.axis}: ${context.values}`
                    : 'Multiple inputs have incompatible shapes',
                actionHint: 'Ensure all inputs have compatible dimensions for merge operation',
            };
        case ValidationCode.AUTO_FLATTEN_APPLIED:
            return {
                title: 'Auto-Flatten Applied',
                description: context?.from
                    ? `Input auto-flattened from rank ${context.from} to ${context.to}`
                    : 'Input was automatically flattened',
                actionHint: 'Disable auto_flatten to preserve original rank',
            };
        case ValidationCode.TARGET_PREDICTION_MISMATCH:
            return {
                title: 'Prediction-Target Mismatch',
                description: 'Prediction and target shapes are incompatible for this loss function',
                actionHint: 'Check loss function requirements and adjust shapes accordingly',
            };
        case ValidationCode.LOSS_INPUT_MISSING:
            return {
                title: 'Loss Input Missing',
                description: 'Loss node requires both prediction and target inputs',
                actionHint: 'Connect both prediction and target/label inputs',
            };
        case ValidationCode.CONNECTION_INVALID:
            return {
                title: 'Invalid Connection',
                description: 'This connection is not allowed',
                actionHint: 'Check port compatibility and connection rules',
            };
        default:
            return {
                title: 'Unknown Error',
                description: `Validation error: ${code}`,
                actionHint: 'Check node configuration and connections',
            };
    }
}
/**
 * Get badge information for a validation state
 */
export function getStateBadgeInfo(state) {
    switch (state) {
        case NodeValidationState.UNCONFIGURED:
            return {
                color: 'gray',
                label: 'Unconfigured',
                icon: 'settings',
                tooltip: 'Node needs configuration',
            };
        case NodeValidationState.AWAITING_INPUT:
            return {
                color: 'blue',
                label: 'Awaiting Input',
                icon: 'arrow-left',
                tooltip: 'Waiting for input connection',
            };
        case NodeValidationState.NEGOTIATING:
            return {
                color: 'yellow',
                label: 'Negotiating',
                icon: 'refresh',
                tooltip: 'Shape has symbolic dimensions or auto-flatten applied',
            };
        case NodeValidationState.ERROR:
            return {
                color: 'red',
                label: 'Error',
                icon: 'alert-circle',
                tooltip: 'Validation error - check connections and configuration',
            };
        case NodeValidationState.VALID:
            return {
                color: 'green',
                label: 'Valid',
                icon: 'check-circle',
                tooltip: 'Node is fully configured and valid',
            };
        default:
            return {
                color: 'gray',
                label: 'Unknown',
                icon: 'help-circle',
                tooltip: 'Unknown state',
            };
    }
}
// =============================================================================
// Pattern Description Formatting
// =============================================================================
/**
 * Format a shape for display
 */
export function formatShape(dims) {
    if (dims.length === 0)
        return 'scalar';
    return `(${dims.join(', ')})`;
}
/**
 * Format inference trace for tooltip
 */
export function formatInferenceTrace(inputShape, outputShape, transformation) {
    return `${formatShape(inputShape)} → ${transformation} → ${formatShape(outputShape)}`;
}
// =============================================================================
// Error Taxonomy Examples
// =============================================================================
export const ERROR_EXAMPLES = {
    [ValidationCode.OK]: {
        message: '',
        hint: '',
    },
    [ValidationCode.CONFIG_INCOMPLETE]: {
        message: 'Output features not set.',
        hint: 'Enter a positive number.',
    },
    [ValidationCode.INPUT_SHAPE_PENDING]: {
        message: 'Waiting for upstream shape.',
        hint: 'Connect a source node.',
    },
    [ValidationCode.PATTERN_MISMATCH_RANK]: {
        message: 'Conv2D needs rank ≥4; got rank 3.',
        hint: 'Insert Reshape or adjust source.',
    },
    [ValidationCode.PATTERN_MISMATCH_AXIS]: {
        message: 'Axis 1 expects channels dimension.',
        hint: 'Check tensor layout.',
    },
    [ValidationCode.FEATURE_INCOMPATIBLE]: {
        message: 'Linear expects 256 features; got 128.',
        hint: 'Align in_features or add projection.',
    },
    [ValidationCode.SYMBOL_UNRESOLVED]: {
        message: 'Sequence length symbol mismatch (T vs L).',
        hint: 'Unify via config mapping.',
    },
    [ValidationCode.MULTI_INPUT_CONFLICT]: {
        message: 'Concat axis sizes differ: 32 vs 64.',
        hint: 'Select different merge axis or reshape.',
    },
    [ValidationCode.AUTO_FLATTEN_APPLIED]: {
        message: 'Input auto-flattened from rank 3 to 2.',
        hint: 'Disable auto_flatten to preserve rank.',
    },
    [ValidationCode.TARGET_PREDICTION_MISMATCH]: {
        message: 'CrossEntropy expects (B, K) predictions with (B,) targets.',
        hint: 'Adjust prediction or target shapes.',
    },
    [ValidationCode.LOSS_INPUT_MISSING]: {
        message: 'Loss requires prediction and target inputs.',
        hint: 'Connect both inputs.',
    },
    [ValidationCode.CONNECTION_INVALID]: {
        message: 'Cannot connect output to input.',
        hint: 'Check port types.',
    },
};
