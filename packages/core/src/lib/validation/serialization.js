/**
 * Serialization adapters for backend compatibility
 * Handles conversion between extended shape format and backend-expected format
 */
import { isNumeric, isSymbolic } from './matchers';
// =============================================================================
// Conversion Functions
// =============================================================================
/**
 * Convert extended tensor shape to backend-compatible format
 * Replaces symbolic dimensions with -1 (placeholder for unknown)
 */
export function prepareForBackend(shape) {
    const numericDims = shape.dims.map(dim => {
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
export function prepareForExtendedBackend(shape) {
    const numericDims = [];
    const symbolMapping = {};
    const symbolValues = {};
    shape.dims.forEach((dim, index) => {
        if (isNumeric(dim)) {
            numericDims.push(dim);
        }
        else {
            // Symbolic dimension
            numericDims.push(-1);
            symbolValues[index] = dim;
            symbolMapping[dim] = index;
        }
    });
    const result = {
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
export function deserializeFromBackend(backendShape) {
    const dims = backendShape.dims.map((dim, index) => {
        if (dim === -1) {
            // Check if we have symbol information
            const extendedShape = backendShape;
            if (extendedShape.symbolValues && extendedShape.symbolValues[index]) {
                return extendedShape.symbolValues[index];
            }
            // Unknown dimension - use generic symbol
            return `dim${index}`;
        }
        return dim;
    });
    const shape = {
        dims,
        description: backendShape.description,
    };
    // Restore flags if available
    const extendedShape = backendShape;
    if (extendedShape.flags) {
        shape.flags = extendedShape.flags;
    }
    return shape;
}
/**
 * Current serialization format version
 */
export const SERIALIZATION_VERSION = 2;
/**
 * Serialize node data for project storage
 */
export function serializeNodeData(nodeData, useExtendedFormat = true) {
    const serialized = {
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
export function deserializeNodeData(serialized) {
    const nodeData = {
        blockType: serialized.blockType,
        label: serialized.label,
        config: serialized.config,
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
export function migrateOldShape(oldShape) {
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
        const shapeObj = oldShape;
        // Check for dims array
        if (Array.isArray(shapeObj.dims)) {
            return {
                dims: shapeObj.dims.map(d => typeof d === 'number' || typeof d === 'string' ? d : -1),
                description: shapeObj.description,
                flags: shapeObj.flags,
                provenance: shapeObj.provenance,
            };
        }
    }
    return undefined;
}
/**
 * Check if shape needs migration
 */
export function needsMigration(shape) {
    if (!shape)
        return false;
    // Array format needs migration
    if (Array.isArray(shape))
        return true;
    // Check for old object format without version
    if (typeof shape === 'object') {
        const shapeObj = shape;
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
export function isFullySerializable(shape) {
    return shape.dims.every(isNumeric);
}
/**
 * Get list of symbolic dimensions in a shape
 */
export function getSymbolicDimensions(shape) {
    return shape.dims.filter(isSymbolic);
}
/**
 * Resolve symbolic dimensions using a mapping
 */
export function resolveSymbols(shape, symbolValues) {
    const resolvedDims = shape.dims.map(dim => {
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
export function shapeToString(shape) {
    if (shape.dims.length === 0) {
        return 'scalar';
    }
    return `(${shape.dims.join(', ')})`;
}
/**
 * Parse a shape string to TensorShape
 */
export function parseShapeString(shapeStr) {
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
    const dims = dimStrs.map(s => {
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
