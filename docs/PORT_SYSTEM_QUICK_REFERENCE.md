# Port System Quick Reference

## Port Semantic Types

```typescript
enum PortSemantic {
  Data = 'data',              // General data tensor
  Labels = 'labels',          // Ground truth labels/targets
  Predictions = 'predictions', // Model predictions/outputs
  Features = 'features',      // Intermediate feature representations
  Anchor = 'anchor',          // Triplet loss anchor sample
  Positive = 'positive',      // Triplet loss positive sample
  Negative = 'negative',      // Triplet loss negative sample
  Loss = 'loss',              // Loss value output
  Any = 'any',                // Accepts any connection type
  Generic = 'generic'         // Default/unspecified type
}
```

## Compatibility Matrix

| Source ↓ / Target → | Data | Labels | Predictions | Features | Anchor | Positive | Negative | Loss | Any | Generic |
|---------------------|------|--------|-------------|----------|--------|----------|----------|------|-----|---------|
| **Data**            | ✅   | ❌     | ❌          | ✅       | ✅     | ✅       | ✅       | ❌   | ✅  | ✅      |
| **Labels**          | ❌   | ✅     | ❌          | ❌       | ❌     | ❌       | ❌       | ❌   | ✅  | ✅      |
| **Predictions**     | ❌   | ❌     | ✅          | ❌       | ❌     | ❌       | ❌       | ❌   | ✅  | ✅      |
| **Features**        | ✅   | ❌     | ❌          | ✅       | ✅     | ✅       | ✅       | ❌   | ✅  | ✅      |
| **Anchor**          | ✅   | ❌     | ❌          | ✅       | ✅     | ❌       | ❌       | ❌   | ✅  | ✅      |
| **Positive**        | ✅   | ❌     | ❌          | ✅       | ❌     | ✅       | ❌       | ❌   | ✅  | ✅      |
| **Negative**        | ✅   | ❌     | ❌          | ✅       | ❌     | ❌       | ✅       | ❌   | ✅  | ✅      |
| **Loss**            | ❌   | ❌     | ❌          | ❌       | ❌     | ❌       | ❌       | ✅   | ✅  | ✅      |
| **Any**             | ✅   | ✅     | ✅          | ✅       | ✅     | ✅       | ✅       | ✅   | ✅  | ✅      |
| **Generic**         | ✅   | ✅     | ✅          | ✅       | ✅     | ✅       | ✅       | ✅   | ✅  | ✅      |

## Creating Custom Ports

### Frontend (TypeScript)

```typescript
// In your node definition class
import { PortDefinition, PortSemantic } from '@/lib/nodes/ports'

getInputPorts(config: BlockConfig): PortDefinition[] {
  return [
    {
      id: 'input-1',           // Unique ID (used as handleId)
      label: 'Input Data',     // Display name
      semantic: PortSemantic.Data,  // Type for validation
      required: true,          // Optional: must be connected
      description: 'Input tensor data'  // Optional: tooltip text
    }
  ]
}

getOutputPorts(config: BlockConfig): PortDefinition[] {
  return [
    {
      id: 'output-1',
      label: 'Output',
      semantic: PortSemantic.Features
    }
  ]
}
```

### Backend (Python)

```python
# In your NodeSpec
from block_manager.services.nodes.ports import PortDefinition, PortSemantic

CUSTOM_SPEC = NodeSpec(
    # ... other fields ...
    input_ports_config={
        'default': [
            PortDefinition(
                id='input-1',
                label='Input Data',
                semantic=PortSemantic.DATA,
                required=True,
                description='Input tensor data'
            )
        ]
    },
    output_ports_config={
        'default': [
            PortDefinition(
                id='output-1',
                label='Output',
                semantic=PortSemantic.FEATURES
            )
        ]
    }
)
```

## Dynamic Ports Based on Config

```typescript
getInputPorts(config: BlockConfig): PortDefinition[] {
  const mode = config.mode || 'standard'
  
  const portConfigs: Record<string, PortDefinition[]> = {
    'standard': [
      { id: 'input', label: 'Input', semantic: PortSemantic.Data }
    ],
    'advanced': [
      { id: 'input-1', label: 'Input 1', semantic: PortSemantic.Data },
      { id: 'input-2', label: 'Input 2', semantic: PortSemantic.Features }
    ]
  }
  
  return portConfigs[mode] || portConfigs['standard']
}
```

## Handle IDs for BlockNode Rendering

### Loss Node Pattern

```tsx
{inputPorts.map((port, i) => {
  const handleId = `loss-input-${port.id}`  // Prefix + port.id
  
  return (
    <Handle
      type="target"
      position={Position.Left}
      id={handleId}
      // ... styling
    />
  )
})}
```

### DataLoader Pattern

```tsx
// Single outlet
<Handle id="input-output" />

// Multiple outlets
<Handle id={`input-output-${i}`} />

// Ground truth
<Handle id="ground-truth-output" />
```

## Validation Integration

### Check Port Occupancy

```typescript
const isHandleConnected = (handleId: string, isTarget: boolean) => {
  return edges.some(edge => {
    if (isTarget) {
      return edge.target === id && (edge.targetHandle || 'default') === handleId
    } else {
      return edge.source === id && (edge.sourceHandle || 'default') === handleId
    }
  })
}
```

### Connection Validation

```typescript
validateConnection: (connection) => {
  const sourcePort = sourceNodeDef.getOutputPorts(sourceConfig)
    .find(p => p.id === (connection.sourceHandle || 'default'))
  
  const targetPort = targetNodeDef.getInputPorts(targetConfig)
    .find(p => p.id === (connection.targetHandle || 'default'))
  
  if (!sourcePort || !targetPort) return false
  
  return arePortsCompatible(sourcePort, targetPort)
}
```

## Common Patterns

### Single Input/Output (Default)

```typescript
// No need to override - base class provides defaults
// Automatically gets 'default' handle with PortSemantic.Any
```

### Multiple Named Inputs

```typescript
getInputPorts(config: BlockConfig): PortDefinition[] {
  return [
    { id: 'main', label: 'Main Input', semantic: PortSemantic.Data },
    { id: 'auxiliary', label: 'Auxiliary', semantic: PortSemantic.Features }
  ]
}
```

### Config-Dependent Ports

```typescript
getOutputPorts(config: BlockConfig): PortDefinition[] {
  const ports: PortDefinition[] = []
  
  // Always have main output
  ports.push({
    id: 'output',
    label: 'Output',
    semantic: PortSemantic.Data
  })
  
  // Conditional additional output
  if (config.include_attention_weights) {
    ports.push({
      id: 'attention',
      label: 'Attention Weights',
      semantic: PortSemantic.Features
    })
  }
  
  return ports
}
```

### Array of Similar Ports

```typescript
getInputPorts(config: BlockConfig): PortDefinition[] {
  const numInputs = Number(config.num_inputs || 2)
  const ports: PortDefinition[] = []
  
  for (let i = 0; i < numInputs; i++) {
    ports.push({
      id: `input-${i}`,
      label: `Input ${i + 1}`,
      semantic: PortSemantic.Data
    })
  }
  
  return ports
}
```

## Visual Styling

### Connected Port Indicators

```tsx
const isConnected = isHandleConnected(handleId, true)

<Handle
  className={`base-classes ${isConnected ? 'ring-2 ring-offset-1 ring-green-400' : ''}`}
  style={{
    backgroundColor: isConnected ? '#10b981' : normalColor,
    opacity: isConnected ? 1 : 0.8
  }}
/>

<span className={isConnected ? 'opacity-60' : ''}>
  {port.label} {isConnected && '✓'}
</span>
```

### Color Coding by Semantic Type

```typescript
const getPortColor = (semantic: PortSemantic): string => {
  const colors: Record<PortSemantic, string> = {
    [PortSemantic.Data]: '#3b82f6',      // Blue
    [PortSemantic.Labels]: '#10b981',    // Green
    [PortSemantic.Predictions]: '#f59e0b', // Orange
    [PortSemantic.Features]: '#8b5cf6',  // Purple
    [PortSemantic.Anchor]: '#ef4444',    // Red
    [PortSemantic.Positive]: '#10b981',  // Green
    [PortSemantic.Negative]: '#f59e0b',  // Orange
    [PortSemantic.Loss]: '#dc2626',      // Dark red
    [PortSemantic.Any]: '#6b7280',       // Gray
    [PortSemantic.Generic]: '#9ca3af'    // Light gray
  }
  
  return colors[semantic] || colors[PortSemantic.Generic]
}
```

## Error Messages

### Semantic Mismatch

```
"Port semantic mismatch: predictions -> labels"
```

### Port Occupancy

```
"Target handle y_true already connected"
```

### Input Count

```
"Loss function 'triplet_margin' only accepts 3 inputs (Anchor, Positive, Negative). Cannot add more."
```

### Missing Ports

```
"Loss node missing connections to: Positive, Negative"
```

## Testing Checklist

- [ ] Port definitions return correct count
- [ ] Port IDs are unique per node
- [ ] Semantic types match expected connections
- [ ] Handle IDs in BlockNode match port.id
- [ ] Connection validation blocks invalid semantics
- [ ] Duplicate connections prevented
- [ ] Visual feedback shows connected ports
- [ ] Backend validation mirrors frontend logic
- [ ] Config changes update ports correctly
- [ ] Error messages are specific and helpful

## Troubleshooting

### "Property 'getInputPorts' does not exist"

**Solution:** Ensure node class extends `NodeDefinition` base class and `INodeDefinition` interface includes port methods.

### Connections not validating semantics

**Solution:** Check that `arePortsCompatible()` is called in `validateConnection` and port definitions have correct semantic types.

### Visual indicators not showing

**Solution:** Verify `edges` is imported from store and `isHandleConnected()` uses correct handleId format.

### Backend validation failing

**Solution:** Ensure `input_ports_config` in NodeSpec matches frontend port definitions exactly.

## Best Practices

1. **Always use semantic types** - Don't rely on Generic/Any unless truly needed
2. **Make port IDs descriptive** - Use names like 'y_pred', 'anchor', not 'input1', 'input2'
3. **Provide descriptions** - Help users understand what each port expects
4. **Test with real connections** - Verify semantic validation works as intended
5. **Keep frontend-backend in sync** - Port definitions should match between TS and Python
6. **Document custom ports** - Add comments explaining special port configurations
7. **Use required flag** - Mark critical ports that must be connected
8. **Handle config changes** - Ensure ports update when relevant config changes

## Resources

- Full Implementation: [PORT_BASED_CONNECTION_SYSTEM.md](./PORT_BASED_CONNECTION_SYSTEM.md)
- Loss Node Example: [LOSS_NODE_MULTIPLE_INPUTS.md](./LOSS_NODE_MULTIPLE_INPUTS.md)
- Node Architecture: [NODE_DEFINITION_ARCHITECTURE.md](./NODE_DEFINITION_ARCHITECTURE.md)
