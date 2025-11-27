/**
 * PyTorch BatchNorm Layer Node Definition
 * Enhanced with pattern-based validation
 */
import { NodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { TensorShape, BlockConfig, ConfigField, BlockType, ShapePattern } from '../../../types';
export declare class BatchNormNode extends NodeDefinition {
    readonly metadata: NodeMetadata;
    /**
     * Input pattern: 2D for BatchNorm1d or 4D for BatchNorm2d
     */
    readonly inputPattern: ShapePattern;
    /**
     * Output pattern: same as input (pass-through)
     */
    readonly outputPattern: ShapePattern;
    readonly configSchema: ConfigField[];
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
    validateIncomingConnection(sourceNodeType: BlockType, sourceOutputShape: TensorShape | undefined, targetConfig: BlockConfig): string | undefined;
    getDefaultConfig(): BlockConfig;
}
//# sourceMappingURL=batchnorm.d.ts.map