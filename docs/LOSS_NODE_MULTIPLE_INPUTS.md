# Loss Node Multiple Inputs Implementation

## Overview
The Loss block now supports multiple named input ports that vary based on the selected loss function type. This enables proper modeling of loss functions that require different numbers and types of inputs.

## Changes Made

### 1. Backend Changes

#### Updated Models (`block_manager/services/nodes/specs/models.py`)
- Added `InputPortSpec` dataclass to define named input ports
- Added `input_ports` field to `NodeSpec` to support configurable input ports

#### Updated Loss Spec (`block_manager/services/nodes/specs/pytorch/__init__.py`)
- Added `allows_multiple_inputs=True` to LOSS_SPEC
- Added more loss function types: Triplet Loss, Contrastive Loss, NLL, KL Divergence
- Defined default input ports: `y_pred` and `y_true`
- Added `metadata.input_ports_config` to map loss types to their required input ports

### 2. Frontend Changes

#### Updated LossNode Definition (`frontend/src/lib/nodes/definitions/pytorch/loss.ts`)
- Added `InputPort` interface for type safety
- Added `getInputPorts(config)` method that returns appropriate input ports based on loss type
- Supports loss functions:
  - **Standard losses** (MSE, MAE, Cross Entropy, BCE, NLL, KL Div): 2 inputs
    - `y_pred`: Model predictions
    - `y_true`: Ground truth labels/values
  - **Triplet Loss**: 3 inputs
    - `anchor`: Anchor embedding
    - `positive`: Positive example embedding
    - `negative`: Negative example embedding
  - **Contrastive Loss**: 3 inputs
    - `input1`: First input embedding
    - `input2`: Second input embedding
    - `label`: Similarity label (1 or -1)

#### Updated BlockNode Component (`frontend/src/components/BlockNode.tsx`)
- Excluded `loss` from single-input-handle nodes
- Added dedicated rendering logic for loss node with multiple named input ports
- Input ports are displayed on the left side with colored labels
- Single output port on the right (loss value) in red
- Ports are evenly spaced vertically similar to DataLoader outlets

#### Updated Store Validation (`frontend/src/lib/store.ts`)
- Added `loss` to nodes that allow multiple inputs
- Added validation to check that loss nodes have the correct number of inputs based on loss type
- Shows helpful error message indicating required inputs and their names

## Usage Examples

### Mean Squared Error (MSE)
```
Output Block → y_pred (Predictions) → Loss Node
DataLoader → y_true (Ground Truth) → Loss Node
```

### Triplet Loss
```
Model Branch 1 → anchor (Anchor) → Loss Node
Model Branch 2 → positive (Positive) → Loss Node
Model Branch 3 → negative (Negative) → Loss Node
```

### Cross Entropy
```
Output Block → y_pred (Predictions) → Loss Node
DataLoader → y_true (Ground Truth) → Loss Node
```

## Visual Representation

The Loss node now displays:
- **Left side**: Multiple colored input ports with labels (e.g., "Predictions", "Ground Truth", "Anchor", "Positive", "Negative")
- **Right side**: Single red output port for the loss value (to connect to optimizer)

Each input port has:
- A unique color for easy identification
- A descriptive label
- A handle ID in the format `loss-input-{port_id}`

## Benefits

1. **Type Safety**: Each input is clearly labeled, reducing connection errors
2. **Flexibility**: Support for various loss functions with different input requirements
3. **Visual Clarity**: Users can easily see what each input represents
4. **Validation**: System validates that the correct number of inputs are connected
5. **Extensibility**: Easy to add new loss functions with custom input requirements

## Future Enhancements

1. Add more specialized loss functions (e.g., Focal Loss, Dice Loss)
2. Support custom loss functions with user-defined input ports
3. Add tooltips showing expected tensor shapes for each input
4. Implement port-specific validation (e.g., ensure y_true comes from DataLoader)
