/**
 * PyTorch ReLU Activation Node Definition
 */
import { PassthroughNodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { ConfigField } from '../../../types';
export declare class ReLUNode extends PassthroughNodeDefinition {
    readonly metadata: NodeMetadata;
    readonly configSchema: ConfigField[];
}
//# sourceMappingURL=relu.d.ts.map