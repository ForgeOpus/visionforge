/**
 * PyTorch Softmax Activation Node Definition
 */
import { PassthroughNodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
export class SoftmaxNode extends PassthroughNodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'softmax',
                label: 'Softmax',
                category: 'activation',
                color: 'var(--color-destructive)',
                icon: 'Percent',
                description: 'Softmax activation',
                framework: BackendFramework.PyTorch
            }
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [
                {
                    name: 'dim',
                    label: 'Dimension',
                    type: 'number',
                    default: -1,
                    description: 'Dimension along which to apply'
                }
            ]
        });
    }
}
