# Neural Network Blocks and Connection Rules

This document describes all available neural network blocks in VisionForge and the rules governing how they can be connected together.

## Block Categories

### Input Layer
Blocks that define the entry points for data into the neural network.

### Basic Layers
Common neural network building blocks for standard architectures.

### Advanced Layers
Specialized blocks for complex architectures (attention, transformers, etc.).

### Merge/Split Layers
Blocks that combine or split multiple tensor streams.

---

## Available Blocks

### Input
**Category:** Input  
**Description:** Define input tensor shape for any modality (text, image, audio, etc.)

**Configuration:**
- **Custom Label** (optional): Custom label for this input node
- **Note** (optional): Notes or comments about this input

**Input Requirements:** Can receive connections only from data source nodes (e.g., DataLoader)

**Output Shape:** Passes through the shape from connected data source, or user-defined if no connection

**Connection Rules:**
- Can receive connections from data source nodes (DataLoader)
- Cannot receive connections from other processing nodes
- Can connect to any processing block

---

### DataLoader
**Category:** Input  
**Description:** Load and prepare input data with optional ground truth labels

**Configuration:**
- **Input Shape** (required): Input tensor dimensions as JSON array
  - Image example: `[1, 3, 224, 224]` - [batch, channels, height, width]
  - Text example: `[32, 512, 768]` - [batch, sequence, embedding]
  - Audio example: `[16, 1, 16000]` - [batch, channels, samples]
  - Tabular example: `[8, 100, 13]` - [batch, rows, features]
- **Include Ground Truth Output** (optional, default: false): Enable a second output for ground truth labels
- **Ground Truth Shape** (optional, default: `[1, 10]`): Shape for ground truth labels when enabled
- **Randomize Data** (optional, default: false): Use random synthetic data for testing
- **CSV File Path** (optional): Path to CSV file for data loading

**Output Shape:** As specified in Input Shape configuration

**Connection Rules:**
- Cannot receive connections (it's a data source)
- Can connect to Input nodes or processing blocks
- Primary data source for the network

---

### Linear (Fully Connected)
**Category:** Basic  
**Description:** Fully connected layer for dense transformations

**Configuration:**
- **Output Features** (required): Number of output features (min: 1)
- **Use Bias** (optional, default: true): Add learnable bias parameter

**Input Requirements:** Requires 2D input `[batch, features]`

**Output Shape:** `[batch, output_features]`

**Connection Rules:**
- Requires 2D input tensor
- If input is 4D (e.g., from Conv2D), insert a Flatten layer first
- Cannot connect directly from Conv2D or MaxPool2D without Flatten

---

### Conv2D
**Category:** Basic  
**Description:** 2D convolutional layer for spatial feature extraction

**Configuration:**
- **Output Channels** (required): Number of output channels (min: 1)
- **Kernel Size** (optional, default: 3): Size of convolving kernel (min: 1)
- **Stride** (optional, default: 1): Stride of convolution (min: 1)
- **Padding** (optional, default: 0): Zero-padding added to both sides (min: 0)
- **Dilation** (optional, default: 1): Spacing between kernel elements (min: 1)

**Input Requirements:** Requires 4D input `[batch, channels, height, width]`

**Output Shape:** `[batch, out_channels, out_height, out_width]`  
where:
- `out_height = floor((height + 2 * padding - kernel) / stride + 1)`
- `out_width = floor((width + 2 * padding - kernel) / stride + 1)`

**Connection Rules:**
- Requires 4D input tensor
- Cannot connect from Linear or Flatten without reshaping

---

### MaxPool2D
**Category:** Basic  
**Description:** 2D max pooling for downsampling spatial dimensions

**Configuration:**
- **Kernel Size** (optional, default: 2): Size of pooling window (min: 1)
- **Stride** (optional, default: 2): Stride of pooling window (min: 1)
- **Padding** (optional, default: 0): Zero-padding added to both sides (min: 0)

**Input Requirements:** Requires 4D input `[batch, channels, height, width]`

**Output Shape:** `[batch, channels, out_height, out_width]`  
where:
- `out_height = floor((height - kernel) / stride + 1)`
- `out_width = floor((width - kernel) / stride + 1)`

**Connection Rules:**
- Requires 4D input tensor
- Same restrictions as Conv2D

---

### BatchNorm
**Category:** Basic  
**Description:** Batch normalization for training stability

**Configuration:**
- **Momentum** (optional, default: 0.1): Momentum for running mean/variance (0-1)
- **Epsilon** (optional, default: 0.00001): Value for numerical stability (min: 0)
- **Affine Transform** (optional, default: true): Learn affine parameters (gamma, beta)

**Input Requirements:** Requires 2D or 4D input

**Output Shape:** Same as input

**Connection Rules:**
- Works with 2D `[batch, features]` or 4D `[batch, channels, height, width]` tensors
- Cannot connect from 3D tensors (use LayerNorm for sequence data)

---

### Dropout
**Category:** Basic  
**Description:** Dropout regularization to prevent overfitting

**Configuration:**
- **Dropout Rate** (optional, default: 0.5): Probability of dropping a unit (0-1)

**Input Requirements:** Any tensor dimension

**Output Shape:** Same as input

**Connection Rules:**
- Dimension-agnostic
- Can connect after any layer

---

### ReLU
**Category:** Basic  
**Description:** Rectified Linear Unit activation function

**Configuration:** No configuration required

**Input Requirements:** Any tensor dimension

**Output Shape:** Same as input

**Connection Rules:**
- Dimension-agnostic
- Can connect after any layer

---

### Softmax
**Category:** Basic  
**Description:** Softmax activation for probability distributions

**Configuration:**
- **Dimension** (optional, default: -1): Dimension along which to apply softmax

**Input Requirements:** Any tensor dimension

**Output Shape:** Same as input

**Connection Rules:**
- Dimension-agnostic
- Typically used as final layer for classification

---

### Flatten
**Category:** Basic  
**Description:** Flatten tensor to 2D for fully connected layers

**Configuration:**
- **Start Dimension** (optional, default: 1): First dimension to flatten (min: 0)

**Input Requirements:** Any tensor dimension

**Output Shape:** `[batch, flattened_features]`

**Connection Rules:**
- Can connect from any layer
- Essential bridge between Conv2D/MaxPool2D and Linear layers

---

### Multi-Head Attention
**Category:** Advanced  
**Description:** Multi-head self-attention mechanism

**Configuration:**
- **Number of Heads** (required, default: 8): Number of attention heads (min: 1)
- **Dropout** (optional, default: 0.1): Attention dropout rate (0-1)

**Input Requirements:** Requires 3D input `[batch, sequence, embedding]`

**Output Shape:** Same as input `[batch, sequence, embedding]`

**Connection Rules:**
- Requires 3D input tensor
- Embed dimension must be divisible by number of heads
- Cannot connect from 2D or 4D tensors without reshaping

---

### Concatenate
**Category:** Merge  
**Description:** Concatenate multiple tensors along a specified dimension

**Configuration:**
- **Dimension** (optional, default: 1): Dimension along which to concatenate

**Input Requirements:** Multiple inputs with compatible shapes

**Output Shape:** Computed based on input shapes and concatenation dimension

**Connection Rules:**
- **Accepts multiple inputs** (only merge block that allows this)
- All inputs must have same number of dimensions
- All dimensions except concatenation dimension must match

---

### Add
**Category:** Merge  
**Description:** Element-wise addition of tensors (for residual connections)

**Configuration:** No configuration required

**Input Requirements:** Multiple inputs with identical shapes

**Output Shape:** Same as input

**Connection Rules:**
- **Accepts multiple inputs** (only merge block that allows this)
- All inputs must have **exactly the same shape**
- Commonly used for residual/skip connections

---

### Custom Layer
**Category:** Advanced  
**Description:** Custom layer with user-defined Python operations

**Configuration:**
- **Layer Name** (required): Identifier for your custom layer
- **Python Code** (required): Custom forward pass implementation
- **Output Shape** (optional): Expected output shape (JSON array)
- **Description** (optional): Brief description of functionality

**Input Requirements:** Flexible (user-defined)

**Output Shape:** As specified in configuration, or matches input if not specified

**Connection Rules:**
- Flexible - validation depends on user-defined code
- Use with caution - ensure output shapes are correct

**Code Editor:**
- Opens in a modal dialog (not sidebar)
- Syntax highlighting for Python
- Input tensor available as variable `x`
- Must return output tensor

**Example Code:**
```python
# Simple pass-through
return x

# Apply custom transformation
return torch.sigmoid(x) * 2.0

# Multi-step processing
x = x.view(x.size(0), -1)
x = torch.relu(x)
return x
```

---

## Connection Rules Summary

### By Dimension Requirement

**2D Input Required `[batch, features]`:**
- Linear

**3D Input Required `[batch, sequence, embedding]`:**
- Multi-Head Attention

**4D Input Required `[batch, channels, height, width]`:**
- Conv2D
- MaxPool2D

**2D or 4D Input:**
- BatchNorm

**Dimension-Agnostic (any input):**
- Dropout
- ReLU
- Softmax
- Flatten (converts to 2D)
- Custom (user-defined)

### Special Rules

**Data Source Nodes (Cannot Receive Connections):**
- DataLoader (primary data source)

**Can Only Receive from Data Sources:**
- Input (can receive from DataLoader and other future data sources)

**Multiple Inputs Allowed:**
- Concatenate (shapes must be compatible for concatenation)
- Add (shapes must be identical)

**Single Input Only:**
- All other blocks

### Common Connection Patterns

#### Image Classification CNN with DataLoader
```
DataLoader → Input → Conv2D → ReLU → MaxPool2D → Conv2D → ReLU → 
MaxPool2D → Flatten → Linear → Dropout → Linear → Softmax
```

#### Simple Image Classification (Direct Connection)
```
Input (4D) → Conv2D → ReLU → MaxPool2D → Conv2D → ReLU → 
MaxPool2D → Flatten → Linear → Dropout → Linear → Softmax
```

#### Residual Connection
```
           ┌─────────────┐
           │             │
Input → Conv2D → ReLU → Conv2D → Add → ReLU
                           ↑
                           │
```

#### Multi-Modal Fusion with Separate Data Sources
```
DataLoader (Images) → Input (4D) → Conv2D → Flatten ─┐
                                                      ├→ Concatenate → Linear
DataLoader (Text) → Input (3D) → Attention → Flatten─┘
```

---

## Validation Errors

When attempting invalid connections, you'll see helpful error messages:

- **"Input blocks can only receive connections from data source nodes (DataLoader)"** - Input can't connect from processing blocks
- **"DataLoader blocks cannot receive connections (they are source nodes)"** - DataLoader is a data source
- **"Conv2D requires 4D input, got 2D"** - Need to add reshaping or use different architecture
- **"Linear layer requires 2D input, got 4D. Consider adding a Flatten layer first."** - Insert Flatten between Conv/Pool and Linear
- **"Multi-Head Attention requires 3D input, got 4D"** - Wrong tensor dimensionality
- **"BatchNorm requires 2D or 4D input, got 3D"** - Use LayerNorm for sequences
- **Single input blocks rejecting second connection** - Use Concatenate or Add for multi-input

---

## Tips for Building Architectures

1. **Use DataLoader for complete data pipelines** - Connect DataLoader to Input for proper data flow
2. **Input blocks as shape adapters** - When using DataLoader, Input acts as a shape passthrough
3. **Always define your data shape** - Either in DataLoader or Input block
4. **Use Flatten between Conv/Pool and Linear** - Bridge between spatial and dense layers
5. **Check dimension compatibility** - Hover over blocks to see input/output shapes
6. **Use Add for skip connections** - Perfect for ResNet-style architectures
7. **Concatenate for multi-path fusion** - Combine features from different branches
8. **Custom blocks for experiments** - Quick prototyping without modifying codebase
9. **Dropout for regularization** - Add before Linear layers to prevent overfitting
10. **BatchNorm after Conv/Linear** - Helps training stability

---

## Color Coding

Blocks are color-coded by category for easy identification:
- **Teal**: Input/Output operations
- **Deep Blue**: Basic processing layers
- **Purple**: Advanced layers (Conv2D, Attention)
- **Cyan**: Merge operations
- **Red/Orange**: Activations and regularization
