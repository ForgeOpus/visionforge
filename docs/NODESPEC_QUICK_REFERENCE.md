# NodeSpec System - Quick Reference

## For Backend Developers

### Adding a New Node Type

1. **Create the spec in the appropriate framework module:**

```python
# In block_manager/services/nodes/specs/pytorch/__init__.py (or tensorflow/)

MY_NEW_NODE_SPEC = NodeSpec(
    type="my_node",                    # Unique identifier
    label="My Custom Node",            # Display name
    category="basic",                  # Category for palette
    color="var(--color-primary)",     # CSS color
    icon="Star",                       # Phosphor icon name
    description="Does something cool", # Tooltip text
    framework=Framework.PYTORCH,       # or Framework.TENSORFLOW
    
    config_schema=(                    # Tuple of config fields
        ConfigFieldSpec(
            name="param1",
            label="Parameter 1",
            field_type="number",       # "text", "number", "boolean", "select"
            required=True,
            min=1,
            description="What this param does"
        ),
        ConfigFieldSpec(
            name="activation",
            label="Activation",
            field_type="select",
            default="relu",
            options=(
                ConfigOptionSpec(value="relu", label="ReLU"),
                ConfigOptionSpec(value="tanh", label="Tanh"),
            ),
        ),
    ),
    
    template=NodeTemplateSpec(
        name="pytorch_my_node",
        engine="jinja2",
        content="""nn.MyNode({{ config.param1 }}, activation='{{ config.activation }}')"""
    ),
)
```

2. **Add to NODE_SPECS tuple at bottom of file:**

```python
NODE_SPECS = (
    INPUT_SPEC,
    LINEAR_SPEC,
    MY_NEW_NODE_SPEC,  # <-- Add here
    # ... rest
)
```

3. **No restart needed** - registry uses lazy loading and will pick up changes on next request.

### Creating Shape Functions

```python
# In block_manager/services/nodes/rules/shape.py

def compute_my_node_output(
    input_shape: Optional[TensorShape],
    config: Dict[str, int],
    framework: Framework,
) -> Optional[TensorShape]:
    """Compute output shape for my custom node."""
    if not input_shape:
        return None
    
    dims = input_shape.get("dims", [])
    # Your shape logic here
    
    if framework is Framework.PYTORCH:
        # NCHW format logic
        pass
    else:  # TensorFlow
        # NHWC format logic
        pass
    
    return TensorShape({
        "dims": [batch, ...],
        "description": "Transformed shape"
    })
```

### Creating Validation Functions

```python
# In block_manager/services/nodes/rules/validation.py

def validate_my_node_connection(
    source_spec: NodeSpec,
    target_spec: NodeSpec,
    source_output_shape: Optional[TensorShape],
) -> tuple[bool, Optional[str]]:
    """Custom validation for my node."""
    if source_output_shape:
        dims = source_output_shape.get("dims", [])
        if len(dims) != 4:
            return False, "My node requires 4D input"
    return True, None
```

---

## For Frontend Developers

### Fetching Node Specs

```typescript
import { useNodeSpecs } from '@/lib/useNodeSpecs'

function MyComponent() {
  const { specs, loading, error, getSpec, renderCode } = useNodeSpecs({
    framework: 'pytorch',
    autoFetch: true  // Automatically fetch on mount
  })
  
  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error}</div>
  
  // Get specific spec
  const convSpec = getSpec('conv2d')
  
  // Render code for a node
  const handleRender = async () => {
    const code = await renderCode('conv2d', {
      out_channels: 64,
      kernel_size: 3
    })
    console.log(code)  // "nn.Conv2d(...)"
  }
  
  return (
    <div>
      {specs.map(spec => (
        <div key={spec.type}>{spec.label}</div>
      ))}
    </div>
  )
}
```

### Displaying Code Preview

```typescript
import { CodePreview } from '@/components/CodePreview'

function NodeConfigPanel({ node }) {
  return (
    <div>
      <h3>Configuration</h3>
      {/* Config form here */}
      
      <CodePreview 
        nodeType={node.type}
        framework="pytorch"
        config={node.config}
      />
    </div>
  )
}
```

### Using the API Directly

```typescript
import { getNodeDefinitions, renderNodeCode } from '@/lib/api'

// Get all specs
const response = await getNodeDefinitions('pytorch')
if (response.success) {
  const specs = response.data.definitions
}

// Render code
const codeResponse = await renderNodeCode(
  'conv2d',
  'pytorch',
  { out_channels: 64, kernel_size: 3 }
)
if (codeResponse.success) {
  console.log(codeResponse.data.code)
}
```

### TypeScript Types

```typescript
import type { 
  NodeSpec, 
  ConfigField, 
  Framework 
} from '@/lib/nodeSpec.types'

const spec: NodeSpec = {
  type: 'conv2d',
  label: 'Conv2D',
  category: 'basic',
  framework: 'pytorch',
  config_schema: [
    {
      name: 'out_channels',
      label: 'Output Channels',
      field_type: 'number',
      required: true
    }
  ],
  // ... rest
}
```

---

## Common Patterns

### Template Syntax

```jinja2
{# Access config values #}
{{ config.out_channels }}

{# Conditionals #}
{% if config.use_bias %}bias=True{% else %}bias=False{% endif %}

{# Boolean to lowercase #}
{{ config.use_bias|lower }}

{# Default values #}
{{ config.stride|default(1) }}

{# Loops #}
{% for item in config.layers %}
  Layer {{ loop.index }}: {{ item }}
{% endfor %}
```

### Config Schema Field Types

| field_type | Description | Example Use |
|------------|-------------|-------------|
| `text` | String input | Layer names, custom code |
| `number` | Integer/float input | Dimensions, learning rate |
| `boolean` | True/false checkbox | use_bias, training |
| `select` | Dropdown menu | Activation functions, padding modes |

### Validation Patterns

```python
# Check required field
if field_spec.required and value is None:
    errors.append(f"'{field_spec.label}' is required")

# Check min/max for numbers
if field_spec.min is not None and num_value < field_spec.min:
    errors.append(f"'{field_spec.label}' must be at least {field_spec.min}")

# Check select options
if field_spec.options:
    valid_values = [opt.value for opt in field_spec.options]
    if value not in valid_values:
        errors.append(f"'{field_spec.label}' must be one of: {', '.join(valid_values)}")
```

---

## API Endpoints Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/node-definitions?framework=pytorch` | Get all specs |
| GET | `/api/node-definitions/conv2d?framework=pytorch` | Get single spec |
| POST | `/api/render-node-code` | Render node code |

---

## Debugging Tips

### Backend

```python
# Check what specs are loaded
from block_manager.services.nodes.specs.registry import list_node_specs, Framework
specs = list_node_specs(Framework.PYTORCH)
print(f"Loaded {len(specs)} specs")

# Test template rendering
from block_manager.services.nodes.specs.registry import get_node_spec
from block_manager.services.nodes.templates.renderer import render_node_template

spec = get_node_spec('conv2d', Framework.PYTORCH)
rendered = render_node_template(spec, {'out_channels': 64, 'kernel_size': 3})
print(rendered.code)

# Validate config
from block_manager.services.nodes.rules import validate_config
is_valid, errors = validate_config(spec, config)
if not is_valid:
    print("Errors:", errors)
```

### Frontend

```typescript
// Check API response
const response = await getNodeDefinitions('pytorch')
console.log('Success:', response.success)
console.log('Data:', response.data)
console.log('Error:', response.error)

// Debug rendered code
const codeResponse = await renderNodeCode('conv2d', 'pytorch', config)
console.log('Code:', codeResponse.data.code)
console.log('Context:', codeResponse.data.context)
```

---

## Testing

### Run Backend Tests

```bash
cd project
python ../test_nodespec_system.py
```

### Test Individual Components

```python
# Test spec loading
from block_manager.services.nodes.specs.registry import get_node_spec, Framework
spec = get_node_spec('conv2d', Framework.PYTORCH)
assert spec is not None

# Test serialization
from block_manager.services.nodes.specs.serialization import spec_to_dict
spec_dict = spec_to_dict(spec)
assert 'type' in spec_dict

# Test rendering
from block_manager.services.nodes.templates.renderer import render_node_template
rendered = render_node_template(spec, {'out_channels': 64})
assert 'nn.Conv2d' in rendered.code
```

---

## Performance Considerations

1. **Registry is cached** - Specs loaded once on first access
2. **Frozen dataclasses** - Safe to cache, won't be mutated
3. **Deterministic hashing** - Same spec always produces same hash
4. **Lazy loading** - Specs only loaded when needed

---

## File Locations

| Component | Path |
|-----------|------|
| PyTorch Specs | `block_manager/services/nodes/specs/pytorch/__init__.py` |
| TensorFlow Specs | `block_manager/services/nodes/specs/tensorflow/__init__.py` |
| Spec Models | `block_manager/services/nodes/specs/models.py` |
| Registry | `block_manager/services/nodes/specs/registry.py` |
| Serialization | `block_manager/services/nodes/specs/serialization.py` |
| Template Renderer | `block_manager/services/nodes/templates/renderer.py` |
| Shape Functions | `block_manager/services/nodes/rules/shape.py` |
| Validation | `block_manager/services/nodes/rules/validation.py` |
| API Views | `block_manager/views/architecture_views.py` |
| Frontend Types | `frontend/src/lib/nodeSpec.types.ts` |
| Frontend Hook | `frontend/src/lib/useNodeSpecs.ts` |
| Code Preview | `frontend/src/components/CodePreview.tsx` |

---

## Common Issues

### Template Rendering Errors

**Problem:** `UndefinedError: 'config' is undefined`  
**Solution:** Ensure you're passing config dict to `render_node_template()`

**Problem:** Template outputs `None` for config values  
**Solution:** Check that config keys match field names in config_schema

### API Returns Empty Definitions

**Problem:** `definitions: []` in API response  
**Solution:** Check that NODE_SPECS tuple includes your spec at bottom of file

### Frontend Type Errors

**Problem:** `Property 'config_schema' does not exist`  
**Solution:** Use `configSchema` (camelCase) in frontend, `config_schema` (snake_case) in backend

---

## Next Steps

1. **Add your custom nodes** following the patterns above
2. **Test thoroughly** using the test script
3. **Update frontend** to consume backend specs instead of local definitions
4. **Remove deprecated code** once migration is complete
