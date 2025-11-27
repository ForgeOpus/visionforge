/**
 * ValidationEngine - Core service for node validation and shape inference
 */
import { ValidationCode, InferenceStatus, NodeValidationState, } from './types';
import { matchPattern, areFeaturesCompatible, getRank, isNumeric, isSymbolic, getLastDim, autoFlattenForProjection, flattenShape, canConcatenate, canAdd, computeConcatShape, hasSymbolicDims, } from './matchers';
// =============================================================================
// ValidationEngine Class
// =============================================================================
export class ValidationEngine {
    /**
     * Validate a connection between two nodes
     */
    validateConnection(sourceShape, targetPattern, context) {
        // No source shape yet
        if (!sourceShape || sourceShape.dims.length === 0) {
            return {
                ok: false,
                code: ValidationCode.INPUT_SHAPE_PENDING,
                message: 'Waiting for upstream shape',
                actionHint: 'Connect a source node or configure upstream',
            };
        }
        // Match against pattern
        const matchResult = matchPattern(sourceShape, targetPattern);
        if (!matchResult.matches) {
            return {
                ok: false,
                code: matchResult.code,
                message: matchResult.message,
                actionHint: this.getActionHint(matchResult.code, targetPattern),
            };
        }
        // Check feature compatibility if pattern has endsWith
        if (targetPattern.endsWith?.numeric !== undefined) {
            const lastDim = getLastDim(sourceShape);
            if (lastDim !== undefined) {
                const featureResult = areFeaturesCompatible(lastDim, targetPattern.endsWith.numeric);
                if (!featureResult.matches) {
                    return {
                        ok: false,
                        code: featureResult.code,
                        message: featureResult.message,
                        actionHint: 'Align in_features with upstream output or add projection layer',
                    };
                }
            }
        }
        // Build normalized shape (apply auto-flatten if applicable)
        let normalizedShape = {
            dims: sourceShape.dims,
            description: sourceShape.description,
            flags: sourceShape.flags,
            provenance: sourceShape.provenance,
        };
        // Auto-flatten for projection patterns
        if (context.targetConfig.auto_flatten === true &&
            targetPattern.rank &&
            typeof targetPattern.rank === 'object' &&
            targetPattern.rank.min === 2 &&
            getRank(sourceShape) > 2) {
            const flattened = autoFlattenForProjection(sourceShape);
            normalizedShape = {
                dims: flattened.dims,
                description: flattened.description,
                flags: { ...flattened.flags, autoFlatten: true },
                provenance: {
                    source: 'computed',
                    transformation: 'auto_flatten',
                    fromNodeId: context.sourceNodeId,
                },
            };
            return {
                ok: true,
                code: ValidationCode.AUTO_FLATTEN_APPLIED,
                message: `Input auto-flattened from rank ${getRank(sourceShape)} to 2`,
                normalizedShape,
                actionHint: 'Disable auto_flatten to preserve original rank',
            };
        }
        return {
            ok: true,
            code: ValidationCode.OK,
            message: 'Connection valid',
            normalizedShape,
        };
    }
    /**
     * Infer output shape for a node
     */
    inferOutputShape(nodeType, inputShapes, config, inputPattern, outputPattern, context) {
        // Check if we have required inputs
        if (inputShapes.length === 0) {
            return {
                status: InferenceStatus.PENDING,
                code: ValidationCode.INPUT_SHAPE_PENDING,
                message: 'Waiting for input connection',
                actionHint: 'Connect an upstream node',
            };
        }
        // Validate input shape against pattern
        const inputShape = inputShapes[0];
        const matchResult = matchPattern(inputShape, inputPattern);
        if (!matchResult.matches) {
            return {
                status: InferenceStatus.ERROR,
                code: matchResult.code,
                message: matchResult.message,
                actionHint: this.getActionHint(matchResult.code, inputPattern),
            };
        }
        // Handle auto-flatten
        let effectiveInput = inputShape;
        if (context?.autoFlatten && getRank(inputShape) > 2) {
            effectiveInput = autoFlattenForProjection(inputShape);
        }
        // Compute output based on node type
        const outputShape = this.computeNodeOutput(nodeType, effectiveInput, config, outputPattern);
        if (!outputShape) {
            return {
                status: InferenceStatus.BLOCKED,
                code: ValidationCode.CONFIG_INCOMPLETE,
                message: 'Cannot compute output shape',
                actionHint: 'Check configuration parameters',
            };
        }
        // Determine status based on whether output has symbolic dims
        const status = hasSymbolicDims(outputShape)
            ? InferenceStatus.SYMBOLIC
            : InferenceStatus.RESOLVED;
        return {
            shape: outputShape,
            status,
            code: ValidationCode.OK,
            message: status === InferenceStatus.SYMBOLIC
                ? 'Shape inferred with symbolic dimensions'
                : 'Shape fully resolved',
        };
    }
    /**
     * Negotiate merge of multiple input shapes
     */
    negotiateMerge(inputShapes, mergeMode, axis = 1) {
        if (inputShapes.length === 0) {
            return {
                ok: false,
                code: ValidationCode.INPUT_SHAPE_PENDING,
                message: 'No inputs for merge',
            };
        }
        if (inputShapes.length === 1) {
            return {
                shape: inputShapes[0],
                ok: true,
                code: ValidationCode.OK,
                message: 'Single input - no merge needed',
            };
        }
        switch (mergeMode) {
            case 'concat':
                return this.negotiateConcat(inputShapes, axis);
            case 'add':
                return this.negotiateAdd(inputShapes);
            case 'stack':
                return this.negotiateStack(inputShapes, axis);
            default:
                return {
                    ok: false,
                    code: ValidationCode.CONFIG_INCOMPLETE,
                    message: `Unknown merge mode: ${mergeMode}`,
                };
        }
    }
    /**
     * Compute validation state for a node
     */
    computeNodeState(inputShapes, outputShape, config, requiredInputs = 1) {
        // Check configuration
        if (!this.isConfigComplete(config)) {
            return NodeValidationState.UNCONFIGURED;
        }
        // Check inputs
        if (inputShapes.length < requiredInputs) {
            return NodeValidationState.AWAITING_INPUT;
        }
        // Check output
        if (!outputShape) {
            return NodeValidationState.ERROR;
        }
        // Check for symbolic dimensions
        if (hasSymbolicDims(outputShape)) {
            return NodeValidationState.NEGOTIATING;
        }
        return NodeValidationState.VALID;
    }
    /**
     * Create a complete shape status for a node
     */
    createShapeStatus(inputShapes, outputShape, config, inputValidation, outputInference) {
        return {
            state: this.computeNodeState(inputShapes, outputShape, config),
            inputShapes: inputShapes,
            outputShape: outputShape,
            inputValidation,
            outputInference,
            timestamp: Date.now(),
        };
    }
    // ===========================================================================
    // Private Methods
    // ===========================================================================
    computeNodeOutput(nodeType, inputShape, config, outputPattern) {
        switch (nodeType) {
            case 'linear':
                return this.computeLinearOutput(inputShape, config);
            case 'conv2d':
                return this.computeConv2DOutput(inputShape, config);
            case 'flatten':
                return this.computeFlattenOutput(inputShape, config);
            case 'maxpool2d':
                return this.computePoolOutput(inputShape, config);
            case 'relu':
            case 'softmax':
            case 'dropout':
            case 'batchnorm':
                return inputShape; // Pass-through
            default:
                // For unknown types, return input shape (pass-through)
                return inputShape;
        }
    }
    computeLinearOutput(inputShape, config) {
        const outFeatures = config.out_features;
        if (!outFeatures || outFeatures <= 0) {
            return null;
        }
        // Output has same leading dimensions, with last dimension replaced
        const dims = [...inputShape.dims.slice(0, -1), outFeatures];
        return {
            dims,
            description: `Linear output: ${dims.join('×')}`,
            flags: { inferred: true },
            provenance: {
                source: 'computed',
                transformation: 'linear',
            },
        };
    }
    computeConv2DOutput(inputShape, config) {
        if (getRank(inputShape) < 4)
            return null;
        const outChannels = config.out_channels;
        const kernelSize = config.kernel_size;
        const stride = config.stride || 1;
        const padding = config.padding || 0;
        const dilation = config.dilation || 1;
        if (!outChannels || !kernelSize)
            return null;
        // Get kernel, stride, padding as arrays
        const kH = Array.isArray(kernelSize) ? kernelSize[0] : kernelSize;
        const kW = Array.isArray(kernelSize) ? kernelSize[1] ?? kernelSize[0] : kernelSize;
        const sH = Array.isArray(stride) ? stride[0] : stride;
        const sW = Array.isArray(stride) ? stride[1] ?? stride[0] : stride;
        const pH = Array.isArray(padding) ? padding[0] : padding;
        const pW = Array.isArray(padding) ? padding[1] ?? padding[0] : padding;
        const dH = Array.isArray(dilation) ? dilation[0] : dilation;
        const dW = Array.isArray(dilation) ? dilation[1] ?? dilation[0] : dilation;
        // Compute output spatial dimensions
        const [batch, , inH, inW] = inputShape.dims;
        let outH;
        let outW;
        if (isNumeric(inH)) {
            outH = Math.floor((inH + 2 * pH - dH * (kH - 1) - 1) / sH + 1);
        }
        else {
            outH = `${inH}'`; // Symbolic derived
        }
        if (isNumeric(inW)) {
            outW = Math.floor((inW + 2 * pW - dW * (kW - 1) - 1) / sW + 1);
        }
        else {
            outW = `${inW}'`; // Symbolic derived
        }
        return {
            dims: [batch, outChannels, outH, outW],
            description: `Conv2D output`,
            flags: {
                inferred: true,
                symbolic: isSymbolic(outH) || isSymbolic(outW),
            },
            provenance: {
                source: 'computed',
                transformation: 'conv2d',
            },
        };
    }
    computeFlattenOutput(inputShape, config) {
        const startDim = config.start_dim ?? 1;
        const endDim = config.end_dim ?? -1;
        return flattenShape(inputShape, startDim, endDim);
    }
    computePoolOutput(inputShape, config) {
        if (getRank(inputShape) < 4)
            return null;
        const kernelSize = config.kernel_size;
        const stride = config.stride || kernelSize;
        const padding = config.padding || 0;
        if (!kernelSize)
            return null;
        const kH = Array.isArray(kernelSize) ? kernelSize[0] : kernelSize;
        const kW = Array.isArray(kernelSize) ? kernelSize[1] ?? kernelSize[0] : kernelSize;
        const sH = Array.isArray(stride) ? stride[0] : (typeof stride === 'number' ? stride : kH);
        const sW = Array.isArray(stride) ? stride[1] ?? stride[0] : (typeof stride === 'number' ? stride : kW);
        const pH = Array.isArray(padding) ? padding[0] : padding;
        const pW = Array.isArray(padding) ? padding[1] ?? padding[0] : padding;
        const [batch, channels, inH, inW] = inputShape.dims;
        let outH;
        let outW;
        if (isNumeric(inH)) {
            outH = Math.floor((inH + 2 * pH - kH) / sH + 1);
        }
        else {
            outH = `${inH}'`;
        }
        if (isNumeric(inW)) {
            outW = Math.floor((inW + 2 * pW - kW) / sW + 1);
        }
        else {
            outW = `${inW}'`;
        }
        return {
            dims: [batch, channels, outH, outW],
            description: 'Pool output',
            flags: {
                inferred: true,
                symbolic: isSymbolic(outH) || isSymbolic(outW),
            },
            provenance: {
                source: 'computed',
                transformation: 'maxpool2d',
            },
        };
    }
    negotiateConcat(inputShapes, axis) {
        const canConcat = canConcatenate(inputShapes, axis);
        if (!canConcat.matches) {
            const conflicts = this.findMergeConflicts(inputShapes, axis);
            return {
                ok: false,
                code: canConcat.code,
                message: canConcat.message,
                conflicts,
            };
        }
        const shape = computeConcatShape(inputShapes, axis);
        if (!shape) {
            return {
                ok: false,
                code: ValidationCode.MULTI_INPUT_CONFLICT,
                message: 'Failed to compute concatenation result',
            };
        }
        return {
            shape: shape,
            ok: true,
            code: ValidationCode.OK,
            message: `Concatenated along axis ${axis}`,
        };
    }
    negotiateAdd(inputShapes) {
        const canAddResult = canAdd(inputShapes);
        if (!canAddResult.matches) {
            const conflicts = this.findMergeConflicts(inputShapes, -1);
            return {
                ok: false,
                code: canAddResult.code,
                message: canAddResult.message,
                conflicts,
            };
        }
        // Output is same as input
        return {
            shape: inputShapes[0],
            ok: true,
            code: ValidationCode.OK,
            message: 'Element-wise addition',
        };
    }
    negotiateStack(inputShapes, axis) {
        // All shapes must be identical
        const canAddResult = canAdd(inputShapes);
        if (!canAddResult.matches) {
            const conflicts = this.findMergeConflicts(inputShapes, -1);
            return {
                ok: false,
                code: canAddResult.code,
                message: canAddResult.message,
                conflicts,
            };
        }
        // Insert new dimension at axis
        const baseDims = inputShapes[0].dims;
        const newDims = [
            ...baseDims.slice(0, axis),
            inputShapes.length,
            ...baseDims.slice(axis),
        ];
        return {
            shape: {
                dims: newDims,
                description: `Stacked ${inputShapes.length} tensors at axis ${axis}`,
                flags: { inferred: true },
                provenance: {
                    source: 'computed',
                    transformation: 'stack',
                },
            },
            ok: true,
            code: ValidationCode.OK,
            message: `Stacked along new axis ${axis}`,
        };
    }
    findMergeConflicts(inputShapes, ignoreAxis) {
        const conflicts = [];
        if (inputShapes.length < 2)
            return conflicts;
        const refShape = inputShapes[0];
        const refRank = getRank(refShape);
        for (let d = 0; d < refRank; d++) {
            if (d === ignoreAxis)
                continue;
            const values = [];
            const nodeIds = [];
            for (let i = 0; i < inputShapes.length; i++) {
                if (i >= inputShapes.length)
                    continue;
                const shape = inputShapes[i];
                if (d < shape.dims.length) {
                    values.push(shape.dims[d]);
                    nodeIds.push(`input_${i}`);
                }
            }
            // Check for conflicts
            const numericValues = values.filter(isNumeric);
            if (numericValues.length > 1) {
                const unique = new Set(numericValues);
                if (unique.size > 1) {
                    conflicts.push({
                        axis: d,
                        values,
                        sourceNodeIds: nodeIds,
                        description: `Axis ${d}: values differ (${values.join(' vs ')})`,
                    });
                }
            }
        }
        return conflicts;
    }
    isConfigComplete(config) {
        // Basic check - could be enhanced with schema validation
        return Object.keys(config).length > 0;
    }
    getActionHint(code, pattern) {
        switch (code) {
            case ValidationCode.PATTERN_MISMATCH_RANK:
                if (pattern.rank) {
                    const rankSpec = typeof pattern.rank === 'number'
                        ? pattern.rank
                        : pattern.rank.min ?? pattern.rank.exact;
                    return `Add Reshape/Flatten to get ${rankSpec}D tensor`;
                }
                return 'Adjust tensor rank to match requirements';
            case ValidationCode.FEATURE_INCOMPATIBLE:
                return 'Align feature dimensions or add projection layer';
            case ValidationCode.SYMBOL_UNRESOLVED:
                return 'Resolve symbolic dimensions via upstream configuration';
            default:
                return 'Check node configuration and connections';
        }
    }
}
// Export singleton instance
export const validationEngine = new ValidationEngine();
