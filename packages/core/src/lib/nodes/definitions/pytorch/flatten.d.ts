/**
 * PyTorch Flatten Layer Node Definition
 * Enhanced with pattern-based validation
 */
import { NodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { TensorShape, BlockConfig, ConfigField, ShapePattern } from '../../../types';
export declare class FlattenNode extends NodeDefinition {
    readonly metadata: NodeMetadata;
    /**
     * Input pattern: at least 2D tensor
     */
    readonly inputPattern: ShapePattern;
    readonly configSchema: ConfigField[];
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
    getDefaultConfig(): BlockConfig;
}
//# sourceMappingURL=flatten.d.ts.map