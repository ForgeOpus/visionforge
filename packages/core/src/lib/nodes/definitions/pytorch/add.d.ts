/**
 * PyTorch Add (Element-wise) Node Definition
 * Enhanced with pattern-based validation and conflict reporting
 */
import { MergeNodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { TensorShape, BlockConfig, ConfigField, BlockType, ShapePattern } from '../../../types';
export declare class AddNode extends MergeNodeDefinition {
    readonly metadata: NodeMetadata;
    /**
     * Input pattern: element-wise addition requires same shape
     */
    readonly inputPattern: ShapePattern;
    readonly configSchema: ConfigField[];
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
    /**
     * Special method for computing output shape with multiple inputs
     * All inputs must have the same shape for element-wise addition
     */
    computeMultiInputShape(inputShapes: TensorShape[], config: BlockConfig): TensorShape | undefined;
    /**
     * Validate multiple inputs have matching shapes
     */
    validateMultipleInputs(inputShapes: TensorShape[], config: BlockConfig): string | undefined;
    validateIncomingConnection(sourceNodeType: BlockType, sourceOutputShape: TensorShape | undefined, targetConfig: BlockConfig): string | undefined;
    getDefaultConfig(): BlockConfig;
}
//# sourceMappingURL=add.d.ts.map