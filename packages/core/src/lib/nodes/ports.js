/**
 * Port Definition System for Node Connections
 * Defines typed ports for inputs and outputs with semantic meaning
 */
/**
 * Check if two ports are semantically compatible for connection
 */
export function arePortsCompatible(source, target) {
    // Ground truth/labels can only connect to label inputs
    if (source.semantic === 'labels') {
        return target.semantic === 'labels';
    }
    // Loss output should only connect to optimizer (not implemented yet, allow for now)
    if (source.semantic === 'loss') {
        return true; // Will be restricted when optimizer nodes are added
    }
    // Predictions can connect to loss or other prediction inputs
    if (source.semantic === 'predictions') {
        return ['predictions', 'loss', 'data'].includes(target.semantic);
    }
    // Data outputs can connect to most inputs (data, anchor, positive, negative, predictions)
    if (source.semantic === 'data') {
        return ['data', 'anchor', 'positive', 'negative', 'predictions', 'input1', 'input2'].includes(target.semantic);
    }
    // Generic compatibility - same semantic types can connect
    return source.semantic === target.semantic;
}
/**
 * Validate if a connection between two specific ports is allowed
 */
export function validatePortConnection(sourcePort, targetPort) {
    // Check type compatibility
    if (sourcePort.type !== 'output') {
        return { valid: false, error: 'Source must be an output port' };
    }
    if (targetPort.type !== 'input') {
        return { valid: false, error: 'Target must be an input port' };
    }
    // Check semantic compatibility
    if (!arePortsCompatible(sourcePort, targetPort)) {
        return {
            valid: false,
            error: `Cannot connect ${sourcePort.semantic} to ${targetPort.semantic}`
        };
    }
    return { valid: true };
}
/**
 * Default single input port for standard nodes
 */
export const DEFAULT_INPUT_PORT = {
    id: 'default',
    label: 'Input',
    type: 'input',
    semantic: 'data',
    required: true,
    description: 'Default input port'
};
/**
 * Default single output port for standard nodes
 */
export const DEFAULT_OUTPUT_PORT = {
    id: 'default',
    label: 'Output',
    type: 'output',
    semantic: 'data',
    required: false,
    description: 'Default output port'
};
