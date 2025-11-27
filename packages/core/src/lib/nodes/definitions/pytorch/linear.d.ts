/**
 * PyTorch Linear Layer Node Definition
 * Enhanced with pattern-based validation and auto_flatten support
 */
import { NodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { TensorShape, BlockConfig, ConfigField, BlockType, ShapePattern } from '../../../types';
export declare class LinearNode extends NodeDefinition {
    readonly metadata: NodeMetadata;
    /**
     * Input pattern: accepts any rank ≥2 with features as last dimension
     */
    readonly inputPattern: ShapePattern;
    readonly configSchema: ConfigField[];
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
    validateIncomingConnection(sourceNodeType: BlockType, sourceOutputShape: TensorShape | undefined, targetConfig: BlockConfig): string | undefined;
    /**
     * Validate configuration with feature inference
     */
    validateConfig(config: BlockConfig): string[];
    getDefaultConfig(): BlockConfig;
}
//# sourceMappingURL=linear.d.ts.map