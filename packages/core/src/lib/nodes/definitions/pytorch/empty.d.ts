/**
 * PyTorch Empty/Placeholder Node Definition
 */
import { PassthroughNodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { ConfigField } from '../../../types';
export declare class EmptyNode extends PassthroughNodeDefinition {
    readonly metadata: NodeMetadata;
    readonly configSchema: ConfigField[];
}
//# sourceMappingURL=empty.d.ts.map