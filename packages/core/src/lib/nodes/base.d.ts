/**
 * Abstract base class for node definitions providing default implementations
 * and shared utilities for all node types.
 */
import { INodeDefinition, NodeMetadata, DimensionRequirement } from './contracts';
import { TensorShape, BlockConfig, ConfigField, BlockType, ShapePattern } from '../types';
import { PortDefinition } from './ports';
/**
 * Abstract base class that all node definitions extend
 * Provides common functionality and enforces interface compliance
 */
export declare abstract class NodeDefinition implements INodeDefinition {
    abstract readonly metadata: NodeMetadata;
    abstract readonly configSchema: ConfigField[];
    /**
     * Input shape pattern this node accepts
     * Override in subclasses to define specific shape requirements
     */
    readonly inputPattern?: ShapePattern;
    /**
     * Output shape pattern this node produces
     * Override in subclasses to define output shape characteristics
     */
    readonly outputPattern?: ShapePattern;
    /**
     * Get input ports for this node based on configuration
     * Override in subclasses for nodes with multiple or dynamic inputs
     */
    getInputPorts(config: BlockConfig): PortDefinition[];
    /**
     * Get output ports for this node based on configuration
     * Override in subclasses for nodes with multiple or dynamic outputs
     */
    getOutputPorts(config: BlockConfig): PortDefinition[];
    /**
     * Default implementation: pass through input shape
     * Override in subclasses for custom shape transformation
     */
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
    /**
     * Default implementation: allow connections from any source
     * Override in subclasses for specific validation rules
     */
    validateIncomingConnection(sourceNodeType: BlockType, sourceOutputShape: TensorShape | undefined, targetConfig: BlockConfig): string | undefined;
    /**
     * Default implementation: single input only
     * Override to return true for merge nodes
     */
    allowsMultipleInputs(): boolean;
    /**
     * Default implementation: validate required fields from schema
     * Override to add custom validation logic
     */
    validateConfig(config: BlockConfig): string[];
    /**
     * Generate default configuration from schema
     */
    getDefaultConfig(): BlockConfig;
    /**
     * Helper: Validate input tensor dimensions against requirements
     */
    protected validateDimensions(shape: TensorShape | undefined, requirement: DimensionRequirement): string | undefined;
    /**
     * Helper: Compute 2D convolution output dimensions
     */
    protected computeConv2DOutput(inputHeight: number, inputWidth: number, kernelSize: number, stride: number, padding: number, dilation: number): [number, number];
    /**
     * Helper: Compute 2D pooling output dimensions
     */
    protected computePool2DOutput(inputHeight: number, inputWidth: number, kernelSize: number, stride: number, padding: number): [number, number];
    /**
     * Helper: Parse shape from string representation
     */
    protected parseShapeString(shapeStr: string): number[] | undefined;
    /**
     * Helper: Check if all dimensions in shapes match
     */
    protected shapesMatch(shape1: TensorShape, shape2: TensorShape): boolean;
}
/**
 * Base class for input/source nodes that don't receive connections
 */
export declare abstract class SourceNodeDefinition extends NodeDefinition {
    validateIncomingConnection(sourceNodeType: BlockType, sourceOutputShape: TensorShape | undefined, targetConfig: BlockConfig): string | undefined;
}
/**
 * Base class for output/terminal nodes that accept any input
 */
export declare abstract class TerminalNodeDefinition extends NodeDefinition {
    validateIncomingConnection(sourceNodeType: BlockType, sourceOutputShape: TensorShape | undefined, targetConfig: BlockConfig): string | undefined;
}
/**
 * Base class for merge nodes that accept multiple inputs
 */
export declare abstract class MergeNodeDefinition extends NodeDefinition {
    allowsMultipleInputs(): boolean;
}
/**
 * Base class for passthrough/utility nodes
 */
export declare abstract class PassthroughNodeDefinition extends NodeDefinition {
    computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined;
    validateIncomingConnection(sourceNodeType: BlockType, sourceOutputShape: TensorShape | undefined, targetConfig: BlockConfig): string | undefined;
}
//# sourceMappingURL=base.d.ts.map