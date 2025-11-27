/**
 * PyTorch Conv2D Layer Node Definition
 * Enhanced with pattern-based validation
 */
import { NodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { TensorShape, BlockConfig, ConfigField, BlockType, ShapePattern } from '../../../types';
export declare class Conv2DNode extends NodeDefinition {
    readonly metadata: NodeMetadata;
    /**
     * Input pattern: 4D spatial tensor (B, C, H, W)
     */
    readonly inputPattern: ShapePattern;
    readonly configSchema: ConfigField[];
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
    validateIncomingConnection(sourceNodeType: BlockType, sourceOutputShape: TensorShape | undefined, targetConfig: BlockConfig): string | undefined;
    getDefaultConfig(): BlockConfig;
}
//# sourceMappingURL=conv2d.d.ts.map