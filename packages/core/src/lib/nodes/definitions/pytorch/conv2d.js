/**
 * PyTorch Conv2D Layer Node Definition
 * Enhanced with pattern-based validation
 */
import { NodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
import { spatialInput } from '../../../validation/patterns';
import { getRank, isNumeric } from '../../../validation/matchers';
export class Conv2DNode extends NodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'conv2d',
                label: 'Conv2D',
                category: 'basic',
                color: 'var(--color-purple)',
                icon: 'SquareHalf',
                description: '2D convolutional layer',
                framework: BackendFramework.PyTorch
            }
        });
        /**
         * Input pattern: 4D spatial tensor (B, C, H, W)
         */
        Object.defineProperty(this, "inputPattern", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: spatialInput()
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [
                {
                    name: 'in_channels',
                    label: 'Input Channels',
                    type: 'number',
                    required: false,
                    min: 1,
                    description: 'Number of input channels (auto-inferred if not set)'
                },
                {
                    name: 'out_channels',
                    label: 'Output Channels',
                    type: 'number',
                    required: true,
                    min: 1,
                    description: 'Number of output channels'
                },
                {
                    name: 'kernel_size',
                    label: 'Kernel Size',
                    type: 'number',
                    default: 3,
                    min: 1,
                    description: 'Size of convolving kernel'
                },
                {
                    name: 'stride',
                    label: 'Stride',
                    type: 'number',
                    default: 1,
                    min: 1,
                    description: 'Stride of convolution'
                },
                {
                    name: 'padding',
                    label: 'Padding',
                    type: 'number',
                    default: 0,
                    min: 0,
                    description: 'Zero-padding added to both sides'
                },
                {
                    name: 'dilation',
                    label: 'Dilation',
                    type: 'number',
                    default: 1,
                    min: 1,
                    description: 'Spacing between kernel elements'
                }
            ]
        });
    }
    computeOutputShape(inputShape, config) {
        if (!inputShape || !config.out_channels) {
            return undefined;
        }
        const rank = getRank(inputShape);
        if (rank !== 4) {
            return undefined;
        }
        const [batch, , height, width] = inputShape.dims;
        const kernel = config.kernel_size || 3;
        const stride = config.stride || 1;
        const padding = config.padding || 0;
        const dilation = config.dilation || 1;
        const outChannels = config.out_channels;
        // Compute output dimensions
        let outHeight;
        let outWidth;
        if (isNumeric(height)) {
            const [h] = this.computeConv2DOutput(height, 1, kernel, stride, padding, dilation);
            outHeight = h;
        }
        else {
            outHeight = `${height}'`;
        }
        if (isNumeric(width)) {
            const [, w] = this.computeConv2DOutput(1, width, kernel, stride, padding, dilation);
            outWidth = w;
        }
        else {
            outWidth = `${width}'`;
        }
        return {
            dims: [batch, outChannels, outHeight, outWidth],
            description: 'Convolved feature map',
            flags: {
                inferred: true,
                symbolic: !isNumeric(outHeight) || !isNumeric(outWidth)
            },
            provenance: {
                source: 'computed',
                transformation: 'conv2d'
            }
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
        if (!sourceOutputShape) {
            return undefined;
        }
        const rank = getRank(sourceOutputShape);
        if (rank !== 4) {
            return `Conv2D requires 4D input (B, C, H, W), got ${rank}D`;
        }
        // Check input channels if specified
        const inChannels = targetConfig.in_channels;
        if (inChannels && inChannels > 0) {
            const inputChannels = sourceOutputShape.dims[1];
            if (isNumeric(inputChannels) && inputChannels !== inChannels) {
                return `Channel mismatch: input has ${inputChannels} channels, expected ${inChannels}`;
            }
        }
        return undefined;
    }
    getDefaultConfig() {
        return {
            out_channels: '',
            kernel_size: 3,
            stride: 1,
            padding: 0,
            dilation: 1
        };
    }
}
