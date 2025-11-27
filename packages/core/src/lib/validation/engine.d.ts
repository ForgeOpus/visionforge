/**
 * ValidationEngine - Core service for node validation and shape inference
 */
import type { TensorShape, BlockConfig } from '../types';
import { ValidationResult, InferenceResult, MergeResult, ShapePattern, ValidationContext, InferenceContext, NodeValidationState, NodeShapeStatus } from './types';
export declare class ValidationEngine {
    /**
     * Validate a connection between two nodes
     */
    validateConnection(sourceShape: TensorShape | undefined, targetPattern: ShapePattern, context: ValidationContext): ValidationResult;
    /**
     * Infer output shape for a node
     */
    inferOutputShape(nodeType: string, inputShapes: TensorShape[], config: BlockConfig, inputPattern: ShapePattern, outputPattern?: ShapePattern, context?: InferenceContext): InferenceResult;
    /**
     * Negotiate merge of multiple input shapes
     */
    negotiateMerge(inputShapes: TensorShape[], mergeMode: 'concat' | 'add' | 'stack', axis?: number): MergeResult;
    /**
     * Compute validation state for a node
     */
    computeNodeState(inputShapes: TensorShape[], outputShape: TensorShape | undefined, config: BlockConfig, requiredInputs?: number): NodeValidationState;
    /**
     * Create a complete shape status for a node
     */
    createShapeStatus(inputShapes: TensorShape[], outputShape: TensorShape | undefined, config: BlockConfig, inputValidation?: ValidationResult, outputInference?: InferenceResult): NodeShapeStatus;
    private computeNodeOutput;
    private computeLinearOutput;
    private computeConv2DOutput;
    private computeFlattenOutput;
    private computePoolOutput;
    private negotiateConcat;
    private negotiateAdd;
    private negotiateStack;
    private findMergeConflicts;
    private isConfigComplete;
    private getActionHint;
}
export declare const validationEngine: ValidationEngine;
//# sourceMappingURL=engine.d.ts.map