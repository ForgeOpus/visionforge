/**
 * PyTorch Softmax Activation Node Definition
 */
import { PassthroughNodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { ConfigField } from '../../../types';
export declare class SoftmaxNode extends PassthroughNodeDefinition {
    readonly metadata: NodeMetadata;
    readonly configSchema: ConfigField[];
}
//# sourceMappingURL=softmax.d.ts.map