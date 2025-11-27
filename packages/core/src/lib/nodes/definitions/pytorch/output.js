/**
 * PyTorch Output Node Definition
 */
import { TerminalNodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
export class OutputNode extends TerminalNodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'output',
                label: 'Output',
                category: 'output',
                color: 'var(--color-green)',
                icon: 'Export',
                description: 'Define model output and predictions',
                framework: BackendFramework.PyTorch
            }
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: []
        });
    }
    /**
     * Output node provides predictions that can connect to loss functions
     */
    getOutputPorts(config) {
        return [{
                id: 'predictions-output',
                label: 'Predictions',
                type: 'output',
                semantic: 'predictions',
                required: false,
                description: 'Model predictions/output'
            }];
    }
    computeOutputShape(inputShape, config) {
        return inputShape;
    }
}
