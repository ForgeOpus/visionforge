# JSON Export Format Documentation

## Overview

VisionForge exports neural network architectures in a clean, human-readable JSON format. This format is designed to be:
- **Secure**: No code execution, only configuration data
- **Readable**: Well-structured and easy to understand
- **Portable**: Can be shared, versioned, and imported across projects

## Format Structure

```json
{
  "version": "1.0.0",
  "projectName": "My Neural Network",
  "projectDescription": "A custom architecture for image classification",
  "framework": "pytorch",
  "architecture": {
    "nodes": [...],
    "connections": [...]
  },
  "metadata": {
    "exportedAt": 1704567890123,
    "nodeCount": 8,
    "edgeCount": 7
  }
}
```

## Field Descriptions

### Top-Level Fields

- **version** (string): Format version for compatibility checking
- **projectName** (string): Name of the project
- **projectDescription** (string): Optional description
- **framework** (string): Target framework (`"pytorch"` or `"tensorflow"`)
- **architecture** (object): Contains nodes and connections
- **metadata** (object): Export metadata

### Architecture Object

#### Nodes Array

Each node represents a neural network layer/block:

```json
{
  "id": "node-1704567890123",
  "type": "conv2d",
  "label": "Conv2D Layer",
  "category": "basic",
  "config": {
    "in_channels": 3,
    "out_channels": 64,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1
  },
  "inputShape": {
    "dims": ["batch", 3, 224, 224]
  },
  "outputShape": {
    "dims": ["batch", 64, 224, 224]
  }
}
```

Fields:
- **id**: Unique identifier for the node
- **type**: Block type (input, linear, conv2d, etc.)
- **label**: Display name
- **category**: Block category (input, basic, advanced, merge)
- **config**: Layer-specific configuration parameters
- **inputShape**: Input tensor shape (optional)
- **outputShape**: Output tensor shape (optional)

#### Connections Array

Each connection represents data flow between nodes:

```json
{
  "from": "node-1704567890123",
  "to": "node-1704567891456"
}
```

Fields:
- **from**: Source node ID
- **to**: Target node ID

### Metadata Object

```json
{
  "exportedAt": 1704567890123,
  "nodeCount": 8,
  "edgeCount": 7
}
```

- **exportedAt**: Unix timestamp (milliseconds)
- **nodeCount**: Number of nodes in architecture
- **edgeCount**: Number of connections

## Security Features

### What's Excluded

The export format intentionally excludes:
- ❌ UI positioning data (x, y coordinates)
- ❌ Internal React Flow state
- ❌ Generated PyTorch code
- ❌ Training data or weights
- ❌ Executable code of any kind

### What's Included

Only safe, declarative configuration:
- ✅ Layer types and names
- ✅ Configuration parameters (numeric values, booleans, strings)
- ✅ Tensor shape information
- ✅ Connection topology

### Why This is Secure

1. **No Code Execution**: The format contains only data, no executable code
2. **Validated on Import**: All imported data is validated against known block types
3. **Type-Safe**: TypeScript ensures proper data types
4. **Read-Only**: Import creates new instances, doesn't modify existing code

## Example: Complete Export

```json
{
  "version": "1.0.0",
  "projectName": "Simple CNN Classifier",
  "projectDescription": "Basic image classification architecture",
  "framework": "pytorch",
  "architecture": {
    "nodes": [
      {
        "id": "input-1",
        "type": "input",
        "label": "Input Layer",
        "category": "input",
        "config": {
          "shape": "[\"batch\", 3, 224, 224]"
        },
        "outputShape": {
          "dims": ["batch", 3, 224, 224]
        }
      },
      {
        "id": "conv-1",
        "type": "conv2d",
        "label": "Conv2D",
        "category": "basic",
        "config": {
          "in_channels": 3,
          "out_channels": 64,
          "kernel_size": 3,
          "stride": 1,
          "padding": 1
        },
        "inputShape": {
          "dims": ["batch", 3, 224, 224]
        },
        "outputShape": {
          "dims": ["batch", 64, 224, 224]
        }
      },
      {
        "id": "relu-1",
        "type": "relu",
        "label": "ReLU",
        "category": "basic",
        "config": {},
        "inputShape": {
          "dims": ["batch", 64, 224, 224]
        },
        "outputShape": {
          "dims": ["batch", 64, 224, 224]
        }
      },
      {
        "id": "pool-1",
        "type": "maxpool",
        "label": "MaxPool2D",
        "category": "basic",
        "config": {
          "kernel_size": 2,
          "stride": 2
        },
        "inputShape": {
          "dims": ["batch", 64, 224, 224]
        },
        "outputShape": {
          "dims": ["batch", 64, 112, 112]
        }
      },
      {
        "id": "flatten-1",
        "type": "flatten",
        "label": "Flatten",
        "category": "basic",
        "config": {},
        "inputShape": {
          "dims": ["batch", 64, 112, 112]
        },
        "outputShape": {
          "dims": ["batch", 802816]
        }
      },
      {
        "id": "linear-1",
        "type": "linear",
        "label": "Linear",
        "category": "basic",
        "config": {
          "in_features": 802816,
          "out_features": 1000
        },
        "inputShape": {
          "dims": ["batch", 802816]
        },
        "outputShape": {
          "dims": ["batch", 1000]
        }
      }
    ],
    "connections": [
      { "from": "input-1", "to": "conv-1" },
      { "from": "conv-1", "to": "relu-1" },
      { "from": "relu-1", "to": "pool-1" },
      { "from": "pool-1", "to": "flatten-1" },
      { "from": "flatten-1", "to": "linear-1" }
    ]
  },
  "metadata": {
    "exportedAt": 1704567890123,
    "nodeCount": 6,
    "edgeCount": 5
  }
}
```

## Import Behavior

When importing a JSON file:

1. **Validation**: File is checked for proper structure and version
2. **Node Creation**: Nodes are recreated with configuration
3. **Layout**: Nodes are arranged in a grid pattern
4. **Connections**: Edges are recreated between nodes
5. **Validation**: Architecture is validated for errors
6. **Project**: A new project is created (or current is updated)

## Version Compatibility

- Current version: `1.0.0`
- Future versions will maintain backward compatibility
- Unsupported versions will show a clear error message

## Best Practices

### Exporting
- Use descriptive project names
- Add meaningful descriptions
- Validate before exporting to ensure completeness

### Importing
- Keep backup copies of important architectures
- Review imported architectures before modifying
- Validate after import to check for any issues

### Sharing
- JSON files can be safely shared via email, GitHub, etc.
- No sensitive data is included
- Files can be version-controlled with Git

## Troubleshooting

### Import Errors

**"Invalid export file format"**
- File is not valid JSON
- Required fields are missing
- Check file hasn't been corrupted

**"Unsupported export version"**
- File was created with a newer version
- Update VisionForge to latest version
- Contact support if issue persists

**"Unknown block type"**
- File contains block types not available in current version
- Update VisionForge or remove unsupported blocks

## Migration Guide

If you need to modify the JSON manually:

1. Make a backup copy first
2. Ensure JSON syntax is valid
3. Keep all required fields
4. Match block types to available blocks
5. Validate after re-importing

## Future Enhancements

Planned additions to the format:
- Model metadata (author, tags, license)
- Custom block definitions
- Training configuration
- Dataset specifications
- Performance metrics

---

**Note**: This format is designed for architecture definitions only. For full model deployment (including weights), use PyTorch's native save/load mechanisms.
