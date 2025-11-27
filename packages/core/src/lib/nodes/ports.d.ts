/**
 * Port Definition System for Node Connections
 * Defines typed ports for inputs and outputs with semantic meaning
 */
export type PortSemantic = 'data' | 'labels' | 'loss' | 'predictions' | 'anchor' | 'positive' | 'negative' | 'input1' | 'input2' | 'weights';
export interface PortDefinition {
    id: string;
    label: string;
    type: 'input' | 'output';
    semantic: PortSemantic;
    required: boolean;
    description: string;
    acceptsMultiple?: boolean;
}
export interface NodePortSchema {
    inputs: PortDefinition[];
    outputs: PortDefinition[];
}
/**
 * Check if two ports are semantically compatible for connection
 */
export declare function arePortsCompatible(source: PortDefinition, target: PortDefinition): boolean;
/**
 * Validate if a connection between two specific ports is allowed
 */
export declare function validatePortConnection(sourcePort: PortDefinition, targetPort: PortDefinition): {
    valid: boolean;
    error?: string;
};
/**
 * Default single input port for standard nodes
 */
export declare const DEFAULT_INPUT_PORT: PortDefinition;
/**
 * Default single output port for standard nodes
 */
export declare const DEFAULT_OUTPUT_PORT: PortDefinition;
//# sourceMappingURL=ports.d.ts.map