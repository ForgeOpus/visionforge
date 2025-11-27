/**
 * PyTorch MaxPool2D Layer Node Definition
 */
import { NodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types';
export declare class MaxPool2DNode extends NodeDefinition {
    readonly metadata: NodeMetadata;
    readonly configSchema: ConfigField[];
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
    validateIncomingConnection(sourceNodeType: BlockType, sourceOutputShape: TensorShape | undefined, targetConfig: BlockConfig): string | undefined;
}
//# sourceMappingURL=maxpool.d.ts.map