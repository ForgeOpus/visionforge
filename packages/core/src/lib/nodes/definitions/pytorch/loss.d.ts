/**
 * PyTorch Loss Function Node Definition
 * Enhanced with dual-input validation for prediction-target compatibility
 */
import { NodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { TensorShape, BlockConfig, ConfigField, BlockType, ShapePattern } from '../../../types';
import { PortDefinition } from '../../ports';
export interface InputPort {
    id: string;
    label: string;
    description: string;
}
export declare class LossNode extends NodeDefinition {
    readonly metadata: NodeMetadata;
    /**
     * Input pattern: accepts any shape (validated based on loss type)
     */
    readonly inputPattern: ShapePattern;
    /**
     * Output pattern: scalar loss value
     */
    readonly outputPattern: ShapePattern;
    readonly configSchema: ConfigField[];
    /**
     * Get input ports based on the loss type configuration
     */
    getInputPorts(config: BlockConfig): PortDefinition[];
    /**
     * Get output ports - loss always outputs a single scalar loss value
     */
    getOutputPorts(config: BlockConfig): PortDefinition[];
    /**
     * Loss node accepts multiple inputs but always outputs a scalar loss
     */
    allowsMultipleInputs(): boolean;
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
    validateIncomingConnection(sourceNodeType: BlockType, sourceOutputShape: TensorShape | undefined, targetConfig: BlockConfig): string | undefined;
    /**
     * Validate that prediction and target shapes are compatible for the loss type
     */
    validatePredictionTarget(predictionShape: TensorShape, targetShape: TensorShape, lossType: string): string | undefined;
    /**
     * Validate triplet loss inputs (anchor, positive, negative must match)
     */
    validateTripletInputs(anchor: TensorShape, positive: TensorShape, negative: TensorShape): string | undefined;
    validateConfig(config: BlockConfig): string[];
    getDefaultConfig(): BlockConfig;
}
//# sourceMappingURL=loss.d.ts.map