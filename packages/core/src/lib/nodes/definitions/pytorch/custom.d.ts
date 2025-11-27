/**
 * PyTorch Custom Layer Node Definition
 */
import { NodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { TensorShape, BlockConfig, ConfigField } from '../../../types';
export declare class CustomNode extends NodeDefinition {
    readonly metadata: NodeMetadata;
    readonly configSchema: ConfigField[];
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
    validateIncomingConnection(): string | undefined;
    validateConfig(config: BlockConfig): string[];
}
//# sourceMappingURL=custom.d.ts.map