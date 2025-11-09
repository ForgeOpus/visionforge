# Node Definition Architecture

## Overview

VisionForge uses a **modular, class-based node definition system** that eliminates hard-coded conditionals and provides a highly extensible architecture for adding new neural network layer types. This document describes the architecture, patterns, and procedures for working with node definitions.

## Architecture Principles

### 1. High Decoupling
- Each node type is defined in its own file/module
- No central switch statements or if-else chains
- Validators and shape computation logic live with the node definition

### 2. Framework Agnostic
- Support for multiple backends (PyTorch, TensorFlow)
- Framework-specific implementations when needed
- Shared logic via base classes and mixins

### 3. Automatic Discovery
- Registry pattern with dynamic loading
- No manual registration required
- Add a file → node automatically available

### 4. Non-Breaking Migration
- Legacy adapter maintains backward compatibility
- Gradual migration path
- Deprecation warnings guide developers

## System Components

### Frontend (TypeScript)

```
frontend/src/lib/nodes/
├── contracts.ts          # Interfaces and type definitions
├── base.ts              # Abstract base classes
├── registry.ts          # Auto-discovery and loading
├── definitions/
│   ├── pytorch/         # PyTorch-specific nodes
│   │   ├── linear.ts
│   │   ├── conv2d.ts
│   │   └── ...
│   └── tensorflow/      # TensorFlow-specific nodes
│       ├── linear.ts
│       └── ...
└── legacy/
    └── blockDefinitionsAdapter.ts  # Backward compatibility
```

### Backend (Python)

```
block_manager/services/nodes/
├── __init__.py
├── base.py              # Base classes and mixins
├── registry.py          # Dynamic loading and registration
├── pytorch/             # PyTorch node implementations
│   ├── __init__.py
│   ├── linear.py
│   ├── conv2d.py
│   └── ...
└── tensorflow/          # TensorFlow node implementations
    ├── __init__.py
    └── ...
```

## Key Interfaces

### Frontend: INodeDefinition

```typescript
interface INodeDefinition {
  readonly metadata: NodeMetadata
  readonly configSchema: ConfigField[]
  
  computeOutputShape(
    inputShape: TensorShape | undefined, 
    config: BlockConfig
  ): TensorShape | undefined
  
  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined
  
  allowsMultipleInputs(): boolean
  validateConfig(config: BlockConfig): string[]
  getDefaultConfig(): BlockConfig
}
```

### Backend: NodeDefinition

```python
class NodeDefinition(ABC):
    @property
    @abstractmethod
    def metadata(self) -> NodeMetadata:
        pass
    
    @property
    @abstractmethod
    def config_schema(self) -> List[ConfigField]:
        pass
    
    @abstractmethod
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        pass
    
    def validate_incoming_connection(...) -> Optional[str]:
        pass
    
    def allows_multiple_inputs(self) -> bool:
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        pass
```

## Base Class Hierarchy

### Frontend

- `NodeDefinition` - Abstract base for all nodes
- `SourceNodeDefinition` - Input/source nodes (no incoming connections)
- `TerminalNodeDefinition` - Output/terminal nodes (accept any input)
- `MergeNodeDefinition` - Multi-input nodes (concat, add)
- `PassthroughNodeDefinition` - Nodes that preserve input shape

### Backend

- `NodeDefinition` - Abstract base with mixins
- `ShapeComputerMixin` - Utilities for shape calculation
- `ValidatorMixin` - Common validation patterns
- `SourceNodeDefinition` - Source nodes
- `TerminalNodeDefinition` - Terminal nodes  
- `MergeNodeDefinition` - Multi-input nodes
- `PassthroughNodeDefinition` - Passthrough nodes

## Adding a New Node Type

### Frontend (TypeScript)

1. **Create node file**: `frontend/src/lib/nodes/definitions/pytorch/my_layer.ts`

```typescript
import { NodeDefinition } from '../../base'
import { NodeMetadata, BackendFramework } from '../../contracts'
import { TensorShape, BlockConfig, ConfigField } from '../../../types'

export class MyLayerNode extends NodeDefinition {
  readonly metadata: NodeMetadata = {
    type: 'my_layer',
    label: 'My Layer',
    category: 'basic',
    color: 'var(--color-primary)',
    icon: 'IconName',
    description: 'Description of my layer',
    framework: BackendFramework.PyTorch
  }

  readonly configSchema: ConfigField[] = [
    {
      name: 'param1',
      label: 'Parameter 1',
      type: 'number',
      required: true,
      min: 1,
      description: 'First parameter'
    }
  ]

  computeOutputShape(
    inputShape: TensorShape | undefined,
    config: BlockConfig
  ): TensorShape | undefined {
    // Implement shape computation
    if (!inputShape) return undefined
    return {
      dims: [...inputShape.dims],
      description: 'Output description'
    }
  }

  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined {
    // Implement validation
    return this.validateDimensions(sourceOutputShape, {
      dims: 2,
      description: '[batch, features]'
    })
  }
}
```

2. **Export in index**: Add to `frontend/src/lib/nodes/definitions/pytorch/index.ts`

```typescript
export { MyLayerNode } from './my_layer'
```

3. **Add type**: Update `frontend/src/lib/types.ts`

```typescript
export type BlockType =
  | 'input'
  | 'my_layer'  // Add here
  | ...
```

**Done!** The node is now available throughout the application.

### Backend (Python)

1. **Create node file**: `block_manager/services/nodes/pytorch/my_layer.py`

```python
from typing import Dict, List, Optional, Any
from ..base import NodeDefinition, NodeMetadata, ConfigField, TensorShape, Framework

class MyLayerNode(NodeDefinition):
    """My Custom Layer"""
    
    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            type="my_layer",
            label="My Layer",
            category="basic",
            color="var(--color-primary)",
            icon="IconName",
            description="Description of my layer",
            framework=Framework.PYTORCH
        )
    
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="param1",
                label="Parameter 1",
                type="number",
                required=True,
                min=1,
                description="First parameter"
            )
        ]
    
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        # Implement shape computation
        if not input_shape:
            return None
        return TensorShape(
            dims=list(input_shape.dims),
            description="Output description"
        )
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        # Implement validation
        return self.validate_dimensions(
            source_output_shape,
            2,
            "[batch, features]"
        )
```

2. **Export in package**: Add to `block_manager/services/nodes/pytorch/__init__.py`

```python
from .my_layer import MyLayerNode

__all__ = [
    'MyLayerNode',
    ...
]
```

**Done!** The backend registry will automatically discover and load the node.

## Special Node Types

### Source Nodes (Input, DataLoader)

Use `SourceNodeDefinition` base class - these reject incoming connections:

```typescript
export class InputNode extends SourceNodeDefinition {
  // Automatically rejects incoming connections
}
```

### Terminal Nodes (Output, Loss)

Use `TerminalNodeDefinition` base class - these accept any input:

```typescript
export class OutputNode extends TerminalNodeDefinition {
  // Automatically accepts all connections
}
```

### Merge Nodes (Concat, Add)

Use `MergeNodeDefinition` base class - these allow multiple inputs:

```typescript
export class ConcatNode extends MergeNodeDefinition {
  // Automatically allows multiple inputs
  
  // Implement special multi-input shape computation
  computeMultiInputShape(
    inputShapes: TensorShape[], 
    config: BlockConfig
  ): TensorShape | undefined {
    // Compute output from multiple inputs
  }
}
```

### Passthrough Nodes (ReLU, Dropout)

Use `PassthroughNodeDefinition` base class - input shape = output shape:

```typescript
export class ReLUNode extends PassthroughNodeDefinition {
  // Automatically passes through input shape
}
```

## Registry Usage

### Frontend

```typescript
import { 
  getNodeDefinition, 
  getAllNodeDefinitions, 
  BackendFramework 
} from './lib/nodes/registry'

// Get specific node
const linearDef = getNodeDefinition('linear', BackendFramework.PyTorch)

// Get all nodes for a framework
const allPyTorchNodes = getAllNodeDefinitions(BackendFramework.PyTorch)

// Compute shape
const outputShape = linearDef.computeOutputShape(inputShape, config)

// Validate connection
const error = linearDef.validateIncomingConnection(
  'conv2d',
  sourceShape,
  targetConfig
)
```

### Backend

```python
from block_manager.services.nodes.registry import (
    get_node_definition,
    get_all_node_definitions,
    Framework
)

# Get specific node
linear_def = get_node_definition('linear', Framework.PYTORCH)

# Get all nodes
all_pytorch_nodes = get_all_node_definitions(Framework.PYTORCH)

# Compute shape
output_shape = linear_def.compute_output_shape(input_shape, config)
```

## Migration from Legacy System

### Current State

- ✅ New class-based system fully implemented
- ✅ Legacy adapter provides backward compatibility
- ✅ All existing code continues to work unchanged
- ⏳ Gradual migration to new registry in progress

### Deprecation Timeline

1. **Phase 1 (Current)**: Both systems coexist, deprecation warnings shown
2. **Phase 2**: Components migrated to use registry directly
3. **Phase 3**: Legacy adapter marked for removal
4. **Phase 4**: Legacy adapter removed

### Migration Guide

**Old Code:**
```typescript
import { getBlockDefinition } from './lib/blockDefinitions'
const def = getBlockDefinition('linear')
```

**New Code:**
```typescript
import { getNodeDefinition, BackendFramework } from './lib/nodes/registry'
const def = getNodeDefinition('linear', BackendFramework.PyTorch)
```

## Validation Patterns

### Dimension Validation

```typescript
// Require specific dimension count
return this.validateDimensions(sourceOutputShape, {
  dims: 4,
  description: '[batch, channels, height, width]'
})

// Accept multiple dimension counts
return this.validateDimensions(sourceOutputShape, {
  dims: [2, 4],
  description: '(2D or 4D)'
})

// Accept any dimensions
return this.validateDimensions(sourceOutputShape, {
  dims: 'any',
  description: ''
})
```

### Config Validation

```typescript
validateConfig(config: BlockConfig): string[] {
  const errors = super.validateConfig(config)
  
  // Custom validation logic
  if (config.embed_dim % config.num_heads !== 0) {
    errors.push('Embedding dimension must be divisible by number of heads')
  }
  
  return errors
}
```

## Best Practices

1. **One File Per Node** - Keep node definitions focused and isolated
2. **Use Base Classes** - Leverage provided base classes for common patterns
3. **Document Dimensions** - Clearly document expected input/output shapes
4. **Validate Early** - Catch configuration errors in `validateConfig()`
5. **Test Shape Logic** - Verify shape computation with unit tests
6. **Framework Parity** - Keep frontend and backend definitions aligned

## Testing

### Frontend

```typescript
describe('LinearNode', () => {
  const node = new LinearNode()
  
  it('computes output shape correctly', () => {
    const input = { dims: [32, 128], description: '' }
    const config = { out_features: 64 }
    const output = node.computeOutputShape(input, config)
    
    expect(output?.dims).toEqual([32, 64])
  })
  
  it('validates 2D input requirement', () => {
    const input = { dims: [32, 3, 224, 224], description: '' }
    const error = node.validateIncomingConnection('conv2d', input, {})
    
    expect(error).toContain('requires 2D input')
  })
})
```

### Backend

```python
def test_linear_node_shape_computation():
    node = LinearNode()
    input_shape = TensorShape([32, 128])
    config = {'out_features': 64}
    
    output_shape = node.compute_output_shape(input_shape, config)
    
    assert output_shape.dims == [32, 64]

def test_linear_node_validation():
    node = LinearNode()
    input_shape = TensorShape([32, 3, 224, 224])
    
    error = node.validate_incoming_connection('conv2d', input_shape, {})
    
    assert 'requires 2D input' in error
```

## Troubleshooting

### Node Not Appearing in Palette

1. Check node is exported in `index.ts`
2. Verify `BlockType` includes the type
3. Check console for registry errors
4. Ensure metadata is correct

### Shape Computation Not Working

1. Verify `computeOutputShape()` returns valid `TensorShape`
2. Check input shape is defined
3. Validate config parameters exist
4. Test with unit tests

### Validation Too Strict/Loose

1. Review `validateIncomingConnection()` logic
2. Check dimension requirements
3. Consider source node type exceptions
4. Test edge cases

## Future Enhancements

- **Code Generation**: Add `generateCode()` method to node definitions
- **Visual Customization**: Node-specific rendering hints
- **Advanced Validation**: Cross-node validation rules
- **Performance Metrics**: Built-in FLOP/parameter counting
- **Auto-Documentation**: Generate docs from node definitions

## Questions?

See `NODES_AND_RULES.md` for detailed node-specific rules and connection requirements.
