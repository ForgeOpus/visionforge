/**
 * PyTorch DataLoader Node Definition
 */
import { SourceNodeDefinition } from '../../base';
import { NodeMetadata } from '../../contracts';
import { TensorShape, BlockConfig, ConfigField } from '../../../types';
import { PortDefinition } from '../../ports';
export declare class DataLoaderNode extends SourceNodeDefinition {
    readonly metadata: NodeMetadata;
    readonly configSchema: ConfigField[];
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
    validateConfig(config: BlockConfig): string[];
    /**
     * Get output ports based on configuration
     * DataLoader can have multiple data outputs and optionally a ground truth output
     */
    getOutputPorts(config: BlockConfig): PortDefinition[];
}
//# sourceMappingURL=dataloader.d.ts.map