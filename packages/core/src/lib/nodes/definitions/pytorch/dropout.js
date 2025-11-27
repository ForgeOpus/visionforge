/**
 * PyTorch Dropout Layer Node Definition
 */
import { PassthroughNodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
export class DropoutNode extends PassthroughNodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'dropout',
                label: 'Dropout',
                category: 'basic',
                color: 'var(--color-accent)',
                icon: 'Drop',
                description: 'Dropout regularization layer',
                framework: BackendFramework.PyTorch
            }
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [
                {
                    name: 'p',
                    label: 'Drop Probability',
                    type: 'number',
                    default: 0.5,
                    min: 0,
                    max: 1,
                    description: 'Probability of an element being zeroed'
                },
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
    validateConfig(config) {
        const errors = super.validateConfig(config);
        const p = config.p;
        if (p !== undefined && (p < 0 || p > 1)) {
            errors.push('Drop Probability must be between 0 and 1');
        }
        return errors;
    }
}
