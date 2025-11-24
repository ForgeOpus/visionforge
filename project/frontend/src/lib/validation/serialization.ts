/**
 * Serialization adapters for backend compatibility
 * Handles conversion between extended shape format and backend-expected format
 */

import type { TensorShape } from '../types';
import {
  DimensionValue,
  ShapeFlags,
  ShapeProvenance,
} from './types';
import { isNumeric, isSymbolic } from './matchers';

// =============================================================================
// Backend Shape Format
// =============================================================================

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

// =============================================================================
// Conversion Functions
// =============================================================================

/**
 * Convert extended tensor shape to backend-compatible format
 * Replaces symbolic dimensions with -1 (placeholder for unknown)
 */
export function prepareForBackend(shape: TensorShape): BackendTensorShape {
  const numericDims: number[] = shape.dims.map(dim => {
    if (isNumeric(dim)) {
      return dim;
    }
    // Symbolic dimension - use -1 as placeholder
    return -1;
  });

  return {
    dims: numericDims,
    description: shape.description,
  };
}

/**
 * Convert extended tensor shape to extended backend format
 * Preserves symbol information for future backend support
 */
export function prepareForExtendedBackend(shape: TensorShape): ExtendedBackendShape {
  const numericDims: number[] = [];
  const symbolMapping: Record<string, number> = {};
  const symbolValues: Record<number, string> = {};

  shape.dims.forEach((dim, index) => {
    if (isNumeric(dim)) {
      numericDims.push(dim);
    } else {
      // Symbolic dimension
      numericDims.push(-1);
      symbolValues[index] = dim;
      symbolMapping[dim] = index;
    }
  });

  const result: ExtendedBackendShape = {
    dims: numericDims,
    description: shape.description,
  };

  // Only include symbol info if there are symbols
  if (Object.keys(symbolMapping).length > 0) {
    result.symbolMapping = symbolMapping;
    result.symbolValues = symbolValues;
  }

  if (shape.flags) {
    result.flags = shape.flags;
  }

  return result;
}

/**
 * Deserialize backend shape to extended format
 */
export function deserializeFromBackend(
  backendShape: BackendTensorShape | ExtendedBackendShape
): TensorShape {
  const dims: DimensionValue[] = backendShape.dims.map((dim, index) => {
    if (dim === -1) {
      // Check if we have symbol information
      const extendedShape = backendShape as ExtendedBackendShape;
      if (extendedShape.symbolValues && extendedShape.symbolValues[index]) {
        return extendedShape.symbolValues[index];
      }
      // Unknown dimension - use generic symbol
      return `dim${index}`;
    }
    return dim;
  });

  const shape: TensorShape = {
    dims,
    description: backendShape.description,
  };

  // Restore flags if available
  const extendedShape = backendShape as ExtendedBackendShape;
  if (extendedShape.flags) {
    shape.flags = extendedShape.flags;
  }

  return shape;
}

// =============================================================================
// Project Serialization
// =============================================================================

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
export const SERIALIZATION_VERSION = 2;

/**
 * Serialize node data for project storage
 */
export function serializeNodeData(
  nodeData: {
    blockType: string;
    label: string;
    config: Record<string, unknown>;
    inputShape?: TensorShape;
    outputShape?: TensorShape;
    category: string;
  },
  useExtendedFormat: boolean = true
): SerializedNodeData {
  const serialized: SerializedNodeData = {
    blockType: nodeData.blockType,
    label: nodeData.label,
    config: nodeData.config,
    category: nodeData.category,
    version: SERIALIZATION_VERSION,
  };

  if (nodeData.inputShape) {
    serialized.inputShape = useExtendedFormat
      ? prepareForExtendedBackend(nodeData.inputShape)
      : prepareForBackend(nodeData.inputShape);
  }

  if (nodeData.outputShape) {
    serialized.outputShape = useExtendedFormat
      ? prepareForExtendedBackend(nodeData.outputShape)
      : prepareForBackend(nodeData.outputShape);
  }

  return serialized;
}

/**
 * Deserialize node data from project storage
 */
export function deserializeNodeData(
  serialized: SerializedNodeData
): {
  blockType: string;
  label: string;
  config: Record<string, unknown>;
  inputShape?: TensorShape;
  outputShape?: TensorShape;
  category: string;
} {
  const nodeData: {
    blockType: string;
    label: string;
    config: Record<string, unknown>;
    inputShape?: TensorShape;
    outputShape?: TensorShape;
    category: string;
  } = {
    blockType: serialized.blockType,
    label: serialized.label,
    config: serialized.config as Record<string, unknown>,
    category: serialized.category,
  };

  if (serialized.inputShape) {
    nodeData.inputShape = deserializeFromBackend(serialized.inputShape);
  }

  if (serialized.outputShape) {
    nodeData.outputShape = deserializeFromBackend(serialized.outputShape);
  }

  return nodeData;
}

// =============================================================================
// Backward Compatibility
// =============================================================================

/**
 * Migrate old shape format to new format
 * Handles shapes from older project versions
 */
export function migrateOldShape(oldShape: unknown): TensorShape | undefined {
  if (!oldShape) {
    return undefined;
  }

  // Handle array format (oldest format)
  if (Array.isArray(oldShape)) {
    return {
      dims: oldShape.map(d => (typeof d === 'number' ? d : -1)),
      description: 'Migrated from old format',
      provenance: {
        source: 'inferred',
        description: 'Migrated from legacy format',
      },
    };
  }

  // Handle object format
  if (typeof oldShape === 'object') {
    const shapeObj = oldShape as Record<string, unknown>;

    // Check for dims array
    if (Array.isArray(shapeObj.dims)) {
      return {
        dims: shapeObj.dims.map(d =>
          typeof d === 'number' || typeof d === 'string' ? d : -1
        ),
        description: shapeObj.description as string | undefined,
        flags: shapeObj.flags as ShapeFlags | undefined,
        provenance: shapeObj.provenance as ShapeProvenance | undefined,
      };
    }
  }

  return undefined;
}

/**
 * Check if shape needs migration
 */
export function needsMigration(shape: unknown): boolean {
  if (!shape) return false;

  // Array format needs migration
  if (Array.isArray(shape)) return true;

  // Check for old object format without version
  if (typeof shape === 'object') {
    const shapeObj = shape as Record<string, unknown>;
    // Old format if it has dims but no provenance
    if (Array.isArray(shapeObj.dims) && !shapeObj.provenance) {
      return true;
    }
  }

  return false;
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Check if a shape can be fully serialized to backend format
 * (i.e., has no symbolic dimensions)
 */
export function isFullySerializable(shape: TensorShape): boolean {
  return shape.dims.every(isNumeric);
}

/**
 * Get list of symbolic dimensions in a shape
 */
export function getSymbolicDimensions(shape: TensorShape): string[] {
  return shape.dims.filter(isSymbolic) as string[];
}

/**
 * Resolve symbolic dimensions using a mapping
 */
export function resolveSymbols(
  shape: TensorShape,
  symbolValues: Record<string, number>
): TensorShape {
  const resolvedDims: DimensionValue[] = shape.dims.map(dim => {
    if (isSymbolic(dim) && symbolValues[dim] !== undefined) {
      return symbolValues[dim];
    }
    return dim;
  });

  return {
    dims: resolvedDims,
    description: shape.description,
    flags: {
      ...shape.flags,
      symbolic: resolvedDims.some(isSymbolic),
    },
    provenance: {
      source: 'computed',
      transformation: 'symbol_resolution',
      description: `Resolved symbols: ${Object.keys(symbolValues).join(', ')}`,
    },
  };
}

/**
 * Create a shape string for display
 */
export function shapeToString(shape: TensorShape): string {
  if (shape.dims.length === 0) {
    return 'scalar';
  }
  return `(${shape.dims.join(', ')})`;
}

/**
 * Parse a shape string to TensorShape
 */
export function parseShapeString(shapeStr: string): TensorShape | undefined {
  // Handle scalar
  if (shapeStr.toLowerCase() === 'scalar') {
    return { dims: [] };
  }

  // Parse (dim1, dim2, ...) format
  const match = shapeStr.match(/^\(([^)]*)\)$/);
  if (!match) {
    return undefined;
  }

  const dimStrs = match[1].split(',').map(s => s.trim());
  const dims: DimensionValue[] = dimStrs.map(s => {
    const num = parseInt(s, 10);
    return isNaN(num) ? s : num;
  });

  return {
    dims,
    flags: {
      symbolic: dims.some(isSymbolic),
    },
  };
}
