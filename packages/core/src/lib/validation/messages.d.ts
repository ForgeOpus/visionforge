/**
 * User-friendly message mapping for validation codes
 */
import { ValidationCode, NodeValidationState } from './types';
export interface ValidationMessage {
    title: string;
    description: string;
    actionHint: string;
}
/**
 * Get user-friendly message for a validation code
 */
export declare function getValidationMessage(code: ValidationCode, context?: Record<string, unknown>): ValidationMessage;
export interface StateBadgeInfo {
    color: string;
    label: string;
    icon: string;
    tooltip: string;
}
/**
 * Get badge information for a validation state
 */
export declare function getStateBadgeInfo(state: NodeValidationState): StateBadgeInfo;
/**
 * Format a shape for display
 */
export declare function formatShape(dims: (number | string)[]): string;
/**
 * Format inference trace for tooltip
 */
export declare function formatInferenceTrace(inputShape: (number | string)[], outputShape: (number | string)[], transformation: string): string;
export declare const ERROR_EXAMPLES: Record<ValidationCode, {
    message: string;
    hint: string;
}>;
//# sourceMappingURL=messages.d.ts.map