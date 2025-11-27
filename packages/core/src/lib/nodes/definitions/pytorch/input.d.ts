/**
 * PyTorch Input Node Definition
 */
import { NodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { TensorShape, BlockConfig, ConfigField, BlockType } from '../../../types';
export declare class InputNode extends NodeDefinition {
    readonly metadata: NodeMetadata;
    readonly configSchema: ConfigField[];
    validateIncomingConnection(sourceNodeType: BlockType, sourceOutputShape: TensorShape | undefined, targetConfig: BlockConfig): string | undefined;
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
}
//# sourceMappingURL=input.d.ts.map