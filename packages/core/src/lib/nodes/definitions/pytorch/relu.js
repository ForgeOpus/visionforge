/**
 * PyTorch ReLU Activation Node Definition
 */
import { PassthroughNodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
export class ReLUNode extends PassthroughNodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'relu',
                label: 'ReLU',
                category: 'activation',
                color: 'var(--color-accent)',
                icon: 'Pulse',
                description: 'ReLU activation function',
                framework: BackendFramework.PyTorch
            }
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [
                {
                    name: 'inplace',
                    label: 'In-place Operation',
                    type: 'boolean',
                    default: false,
                    description: 'Perform operation in-place to save memory'
                }
            ]
        });
    }
}
