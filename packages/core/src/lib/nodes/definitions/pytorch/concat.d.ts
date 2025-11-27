/**
 * PyTorch Concatenate Node Definition
 * Enhanced with pattern-based validation and conflict reporting
 */
import { MergeNodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { TensorShape, BlockConfig, ConfigField, BlockType, ShapePattern } from '../../../types';
export declare class ConcatNode extends MergeNodeDefinition {
    readonly metadata: NodeMetadata;
    /**
     * Input pattern: concat merge along configurable axis
     */
    readonly inputPattern: ShapePattern;
    readonly configSchema: ConfigField[];
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
    /**
     * Special method for computing output shape with multiple inputs
     * This should be called by the store/registry when multiple inputs are available
     */
    computeMultiInputShape(inputShapes: TensorShape[], config: BlockConfig): TensorShape | undefined;
    /**
     * Validate multiple inputs for concatenation
     */
    validateMultipleInputs(inputShapes: TensorShape[], config: BlockConfig): string | undefined;
    validateIncomingConnection(sourceNodeType: BlockType, sourceOutputShape: TensorShape | undefined, targetConfig: BlockConfig): string | undefined;
    getDefaultConfig(): BlockConfig;
}
//# sourceMappingURL=concat.d.ts.map