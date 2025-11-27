/**
 * PyTorch Custom Layer Node Definition
 */
import { NodeDefinition } from '../../base';
import { BackendFramework } from '../../contracts';
export class CustomNode extends NodeDefinition {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {
                type: 'custom',
                label: 'Custom Layer',
                category: 'advanced',
                color: 'var(--color-purple)',
                icon: 'Code',
                description: 'Custom layer with user-defined code',
                framework: BackendFramework.PyTorch
            }
        });
        Object.defineProperty(this, "configSchema", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [
                {
                    name: 'name',
                    label: 'Layer Name',
                    type: 'text',
                    required: true,
                    placeholder: 'my_custom_layer',
                    description: 'Name for your custom layer'
                },
                {
                    name: 'code',
                    label: 'Python Code',
                    type: 'text',
                    default: `"""Custom Layer Template - PyTorch

TODO: Implement your custom layer following PyTorch conventions.
"""
import torch
import torch.nn as nn

class CustomLayer(nn.Module):
    """
    Custom User-Defined Layer

    Shape:
        - Input: [batch, *] (Define your input shape)
        - Output: [batch, *] (Define your output shape)
    """

    def __init__(self):
        """Initialize the custom layer."""
        super(CustomLayer, self).__init__()

        # TODO: Define your layer parameters here
        # Examples:
        # self.linear = nn.Linear(in_features, out_features)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # self.activation = nn.ReLU()

        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the custom layer."""
        # TODO: Implement your forward pass logic here

        # Placeholder: returns input unchanged
        return x`,
                    description: 'Custom forward pass implementation'
                },
                {
                    name: 'output_shape',
                    label: 'Output Shape',
                    type: 'text',
                    placeholder: '[batch, features]',
                    description: 'Expected output shape (optional, leave empty to match input)'
                },
                {
                    name: 'description',
                    label: 'Description',
                    type: 'text',
                    placeholder: 'Describe what this layer does',
                    description: 'Brief description of the layer functionality'
                }
            ]
        });
    }
    computeOutputShape(inputShape, config) {
        // If user specified output shape, use it
        if (config.output_shape) {
            const dims = this.parseShapeString(String(config.output_shape));
            if (dims) {
                return {
                    dims,
                    description: String(config.description || 'Custom output')
                };
            }
        }
        // Otherwise pass through input shape
        return inputShape;
    }
    validateIncomingConnection() {
        // Custom nodes are flexible and accept connections from any source
        return undefined;
    }
    validateConfig(config) {
        const errors = super.validateConfig(config);
        // Validate name format (should be valid Python identifier)
        const name = String(config.name || '');
        if (name && !/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
            errors.push('Layer Name must be a valid Python identifier');
        }
        // Validate output shape format if provided
        if (config.output_shape && config.output_shape !== '') {
            const dims = this.parseShapeString(String(config.output_shape));
            if (!dims) {
                errors.push('Output Shape must be a valid JSON array of positive numbers');
            }
        }
        return errors;
    }
}
