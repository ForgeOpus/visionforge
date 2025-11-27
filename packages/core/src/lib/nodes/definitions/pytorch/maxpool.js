/**
 * PyTorch MaxPool2D Layer Node Definition
 */
import { NodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
export class MaxPool2DNode extends NodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'maxpool2d',
                label: 'MaxPool2D',
                category: 'basic',
                color: 'var(--color-purple)',
                icon: 'SquaresFour',
                description: '2D max pooling layer',
                framework: BackendFramework.PyTorch
            }
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [
                {
                    name: 'kernel_size',
                    label: 'Kernel Size',
                    type: 'number',
                    default: 2,
                    min: 1,
                    description: 'Size of pooling window'
                },
                {
                    name: 'stride',
                    label: 'Stride',
                    type: 'number',
                    default: 2,
                    min: 1,
                    description: 'Stride of pooling window'
                },
                {
                    name: 'padding',
                    label: 'Padding',
                    type: 'number',
                    default: 0,
                    min: 0,
                    description: 'Zero-padding added to both sides'
                }
            ]
        });
    }
    computeOutputShape(inputShape, config) {
        if (!inputShape) {
            return undefined;
        }
        if (inputShape.dims.length !== 4) {
            return undefined;
        }
        const [batch, channels, height, width] = inputShape.dims;
        const kernel = config.kernel_size;
        const stride = config.stride;
        const padding = config.padding;
        const [outHeight, outWidth] = this.computePool2DOutput(height, width, kernel, stride, padding);
        return {
            dims: [batch, channels, outHeight, outWidth],
            description: 'Pooled feature map'
        };
    }
    validateIncomingConnection(sourceNodeType, sourceOutputShape, targetConfig) {
        // Allow connections from input/dataloader without shape validation
        if (sourceNodeType === 'input' || sourceNodeType === 'dataloader') {
            return undefined;
        }
        // Empty and custom nodes are flexible
        if (sourceNodeType === 'empty' || sourceNodeType === 'custom') {
            return undefined;
        }
        // Validate dimension requirement
        return this.validateDimensions(sourceOutputShape, {
            dims: 4,
            description: '[batch, channels, height, width]'
        });
    }
}
