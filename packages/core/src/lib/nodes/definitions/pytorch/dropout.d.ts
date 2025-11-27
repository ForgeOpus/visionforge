/**
 * PyTorch Dropout Layer Node Definition
 */
import { PassthroughNodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { BlockConfig, ConfigField } from '../../../types';
export declare class DropoutNode extends PassthroughNodeDefinition {
    readonly metadata: NodeMetadata;
    readonly configSchema: ConfigField[];
    validateConfig(config: BlockConfig): string[];
}
//# sourceMappingURL=dropout.d.ts.map