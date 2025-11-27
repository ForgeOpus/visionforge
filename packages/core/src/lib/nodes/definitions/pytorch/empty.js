/**
 * PyTorch Empty/Placeholder Node Definition
 */
import { PassthroughNodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
export class EmptyNode extends PassthroughNodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'empty',
                label: 'Empty Node',
                category: 'utility',
                color: 'var(--color-gray)',
                icon: 'Placeholder',
                description: 'Placeholder node',
                framework: BackendFramework.PyTorch
            }
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [
                {
                    name: 'label',
                    label: 'Custom Label',
                    type: 'text',
                    default: 'Empty Node',
                    placeholder: 'Enter custom label...',
                    description: 'Custom label for this node'
                },
                {
                    name: 'note',
                    label: 'Note',
                    type: 'text',
                    placeholder: 'Add notes here...',
                    description: 'Notes or comments about this placeholder'
                }
            ]
        });
    }
}
