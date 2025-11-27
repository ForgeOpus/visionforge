/**
 * PyTorch Output Node Definition
 */
import { TerminalNodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { TensorShape, BlockConfig, ConfigField } from '../../../types';
import { PortDefinition } from '../../ports';
export declare class OutputNode extends TerminalNodeDefinition {
    readonly metadata: NodeMetadata;
    readonly configSchema: ConfigField[];
    /**
     * Output node provides predictions that can connect to loss functions
     */
    getOutputPorts(config: BlockConfig): PortDefinition[];
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
}
//# sourceMappingURL=output.d.ts.map