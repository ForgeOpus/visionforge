/**
 * Serialization adapters for backend compatibility
 * Handles conversion between extended shape format and backend-expected format
 */
import type { TensorShape } from '../types';
import { ShapeFlags } from './types';
/**
 * Shape format expected by the backend (numeric only)
 */
export interface BackendTensorShape {
    /** Dimension values (numeric only, -1 for unknown/symbolic) */
    dims: number[];
    /** Optional description */
    description?: string;
}
/**
 * Extended backend format with symbol mapping (for future backend support)
 */
export interface ExtendedBackendShape extends BackendTensorShape {
    /** Mapping of symbolic names to their positions */
    symbolMapping?: Record<string, number>;
    /** Original symbolic dimension values */
    symbolValues?: Record<number, string>;
    /** Shape flags */
    flags?: ShapeFlags;
}
/**
 * Convert extended tensor shape to backend-compatible format
 * Replaces symbolic dimensions with -1 (placeholder for unknown)
 */
export declare function prepareForBackend(shape: TensorShape): BackendTensorShape;
/**
 * Convert extended tensor shape to extended backend format
 * Preserves symbol information for future backend support
 */
export declare function prepareForExtendedBackend(shape: TensorShape): ExtendedBackendShape;
/**
 * Deserialize backend shape to extended format
 */
export declare function deserializeFromBackend(backendShape: BackendTensorShape | ExtendedBackendShape): TensorShape;
/**
 * Node data format for project serialization
 */
export interface SerializedNodeData {
    blockType: string;
    label: string;
    config: Record<string, unknown>;
    inputShape?: BackendTensorShape | ExtendedBackendShape;
    outputShape?: BackendTensorShape | ExtendedBackendShape;
    category: string;
    /** Version of the serialization format */
    version?: number;
}
/**
 * Current serialization format version
 */
export declare const SERIALIZATION_VERSION = 2;
/**
 * Serialize node data for project storage
 */
export declare function serializeNodeData(nodeData: {
    blockType: string;
    label: string;
    config: Record<string, unknown>;
    inputShape?: TensorShape;
    outputShape?: TensorShape;
    category: string;
}, useExtendedFormat?: boolean): SerializedNodeData;
/**
 * Deserialize node data from project storage
 */
export declare function deserializeNodeData(serialized: SerializedNodeData): {
    blockType: string;
    label: string;
    config: Record<string, unknown>;
    inputShape?: TensorShape;
    outputShape?: TensorShape;
    category: string;
};
/**
 * Migrate old shape format to new format
 * Handles shapes from older project versions
 */
export declare function migrateOldShape(oldShape: unknown): TensorShape | undefined;
/**
 * Check if shape needs migration
 */
export declare function needsMigration(shape: unknown): boolean;
/**
 * Check if a shape can be fully serialized to backend format
 * (i.e., has no symbolic dimensions)
 */
export declare function isFullySerializable(shape: TensorShape): boolean;
/**
 * Get list of symbolic dimensions in a shape
 */
export declare function getSymbolicDimensions(shape: TensorShape): string[];
/**
 * Resolve symbolic dimensions using a mapping
 */
export declare function resolveSymbols(shape: TensorShape, symbolValues: Record<string, number>): TensorShape;
/**
 * Create a shape string for display
 */
export declare function shapeToString(shape: TensorShape): string;
/**
 * Parse a shape string to TensorShape
 */
export declare function parseShapeString(shapeStr: string): TensorShape | undefined;
//# sourceMappingURL=serialization.d.ts.map