# TensorFlow Backend - Quick Reference Guide

## Overview

VisionForge now supports **TensorFlow/Keras** as a backend framework alongside PyTorch. This guide helps you understand the key differences and how to use the TensorFlow backend effectively.

## Quick Start

### 1. Select TensorFlow Framework
When creating a new project, select **TensorFlow** as your framework:
```typescript
{
  "name": "My CNN Project",
  "framework": "tensorflow"  // or "pytorch"
}
```

### 2. Build Your Architecture
- Drag and drop nodes from the palette
- Configure layer parameters in the config panel
- Connect nodes to create your architecture

### 3. Export Code
Click "Export Code" to generate production-ready TensorFlow code.

## Key Differences: PyTorch vs TensorFlow

### Data Format

**TensorFlow uses NHWC (channels_last):**
```python
# TensorFlow: [batch, height, width, channels]
input_shape = [32, 224, 224, 3]  # 32 RGB images at 224x224

# PyTorch: [batch, channels, height, width]  
input_shape = [32, 3, 224, 224]  # Same data, different ordering
```

### Parameter Names

| Layer Type | PyTorch Parameter | TensorFlow Parameter |
|------------|-------------------|----------------------|
| Conv2D     | `out_channels=64` | `filters=64`         |
| Dense      | `out_features=128`| `units=128`          |
| All Conv   | `stride=2`        | `strides=2`          |

### Padding

**TensorFlow uses string-based padding:**
```python
# PyTorch: padding=1
# TensorFlow: padding='same' or 'valid'
```

- **`'valid'`**: No padding (default)
- **`'same'`**: Padding to preserve spatial dimensions (when stride=1)

## Available Nodes (17 Total)

### Input & Data
- **Input** - Define input tensor shape (NHWC format)
- **DataLoader** - Data loading using `tf.keras.utils.PyDataset`

### Convolutional Layers
- **Conv2D** - 2D convolution (`tf.keras.layers.Conv2D`)
- **Conv1D** - 1D convolution for sequences
- **Conv3D** - 3D convolution for video/volumetric data

### Dense Layers
- **Dense** - Fully connected layer (`tf.keras.layers.Dense`)

### Normalization & Regularization
- **BatchNorm2D** - Batch normalization (`tf.keras.layers.BatchNormalization`)
- **Dropout** - Dropout regularization

### Pooling
- **MaxPool2D** - Max pooling
- **AvgPool2D** - Average pooling
- **GlobalAvgPool2D** - Global average pooling (adaptive)

### Utility
- **Flatten** - Flatten multi-dimensional input to 2D

### Recurrent
- **LSTM** - Long Short-Term Memory
- **GRU** - Gated Recurrent Unit

### Embedding
- **Embedding** - Embedding layer for categorical data

### Merge Operations
- **Concat** - Concatenate tensors along an axis
- **Add** - Element-wise addition

## Configuration Examples

### Conv2D Layer
```json
{
  "filters": 64,          // Number of output channels
  "kernel_size": 3,       // 3x3 kernel
  "strides": 1,           // Stride of 1
  "padding": "same",      // Preserve dimensions
  "activation": "relu"    // Built-in activation (optional)
}
```

### Dense Layer
```json
{
  "units": 128,           // Number of neurons
  "activation": "relu",   // Activation function
  "use_bias": true        // Include bias term
}
```

### LSTM Layer
```json
{
  "units": 128,                // Hidden state size
  "return_sequences": false,   // Return only last output
  "dropout": 0.2,              // Input dropout
  "recurrent_dropout": 0.2     // Recurrent dropout
}
```

## Common Shapes (NHWC Format)

### Image Data
```python
# Single RGB image (224x224)
[1, 224, 224, 3]

# Batch of 32 grayscale images (28x28)
[32, 28, 28, 1]

# Batch of 64 RGB images (256x256)
[64, 256, 256, 3]
```

### Sequence Data
```python
# Batch of 32 sequences, 100 time steps, 50 features
[32, 100, 50]
```

### After Flatten
```python
# From: [32, 7, 7, 64] (feature maps)
# To:   [32, 3136]      (7*7*64 = 3136)
```

## Generated Code Structure

### Model File (`model.py`)
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class YourModel(keras.Model):
    def __init__(self):
        super(YourModel, self).__init__()
        # Layers initialized here
        self.layer_0 = layers.Conv2D(32, 3, padding='same')
        self.layer_1 = layers.MaxPooling2D(2)
        # ...
    
    def call(self, inputs, training=None):
        # Forward pass
        x = self.layer_0(inputs)
        x = self.layer_1(x)
        # ...
        return x

def create_model():
    return YourModel()
```

### Training Script (`train.py`)
Complete training pipeline with:
- Model compilation (`optimizer`, `loss`, `metrics`)
- Callbacks (checkpoints, early stopping, LR scheduling)
- Training loop using `model.fit()`
- Model saving

### Dataset Class (`dataset.py`)
```python
class CustomDataset(keras.utils.PyDataset):
    def __len__(self):
        return num_batches
    
    def __getitem__(self, idx):
        # Return batch in NHWC format
        batch_x = ...  # Shape: [batch_size, height, width, channels]
        batch_y = ...
        return batch_x, batch_y
```

## Error Messages

VisionForge provides detailed error messages to help you fix issues:

### Shape Mismatch
```
Error: "Requires 4D input [batch, height, width, channels], got 2D"
Suggestion: "Add a Reshape layer or ensure previous layer outputs correct dimensions"
```

### Missing Configuration
```
Error: "Conv2D layer requires filters parameter"
Suggestion: "Configure the number of output filters in the configuration panel"
```

### Invalid Connection
```
Error: "DataLoader is a source node and cannot accept incoming connections"
```

## Best Practices

### 1. Input Shape
Always specify input shape in **NHWC format**:
```json
{
  "shape": "[32, 224, 224, 3]"  // Correct: NHWC
  // NOT: "[32, 3, 224, 224]"   // Wrong: This is NCHW (PyTorch format)
}
```

### 2. Padding Choice
- Use **`'same'`** when you want to preserve spatial dimensions
- Use **`'valid'`** when you want dimensions to shrink (no padding)

### 3. Activation Functions
You can either:
- Set activation in layer config: `"activation": "relu"`
- Add a separate activation node (for more control)

### 4. Training-Dependent Layers
These layers automatically handle the `training` parameter:
- Dropout
- BatchNormalization

### 5. Data Format Consistency
Ensure your data is in **NHWC format** throughout:
- Input data: `[batch, height, width, channels]`
- After Conv2D: `[batch, new_height, new_width, filters]`
- After Flatten: `[batch, features]`

## Example Architectures

### Simple CNN for MNIST
```
Input [32, 28, 28, 1]
  ↓
Conv2D (filters=32, kernel=3, padding='same')
  ↓
MaxPool2D (pool_size=2)
  ↓
Conv2D (filters=64, kernel=3, padding='same')
  ↓
MaxPool2D (pool_size=2)
  ↓
Flatten
  ↓
Dense (units=128, activation='relu')
  ↓
Dropout (rate=0.5)
  ↓
Dense (units=10, activation='softmax')
```

### ResNet-style Skip Connection
```
Input
  ├─→ Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm ─┐
  │                                                      ↓
  └─────────────────────────────────────────────────→ Add
                                                         ↓
                                                       ReLU
```

## Troubleshooting

### Issue: "Shape mismatch error"
**Solution**: Check that:
1. Input is in NHWC format
2. Previous layer output matches current layer input requirements
3. Flatten is used before Dense layers when needed

### Issue: "Generated code doesn't run"
**Solution**: 
1. Check TensorFlow is installed: `pip install tensorflow`
2. Verify input data is in NHWC format
3. Check that all required parameters are configured

### Issue: "Padding doesn't work as expected"
**Solution**:
- Use `padding='same'` with `strides=1` to preserve dimensions
- Use `padding='valid'` for no padding (dimensions will shrink)

## Migration from PyTorch

If you have a PyTorch architecture and want to convert to TensorFlow:

1. **Transpose data**: NCHW → NHWC
   ```python
   # PyTorch: [batch, channels, height, width]
   # TensorFlow: [batch, height, width, channels]
   tf_data = torch_data.permute(0, 2, 3, 1)  # If using PyTorch
   # Or: tf_data = np.transpose(data, (0, 2, 3, 1))  # If using NumPy
   ```

2. **Update parameter names**: 
   - `out_channels` → `filters`
   - `out_features` → `units`

3. **Convert padding**:
   - `padding=0` → `padding='valid'`
   - `padding=k//2` (for odd k) → `padding='same'`

## Resources

- **TensorFlow Documentation**: https://tensorflow.org/api_docs/python/tf/keras
- **Keras Layers Guide**: https://keras.io/api/layers/
- **PyDataset Documentation**: https://keras.io/api/utils/python_utils/#pydataset-class

## Support

For issues or questions:
1. Check error messages in the UI
2. Review this guide
3. Consult the full documentation in `TENSORFLOW_IMPLEMENTATION_COMPLETE.md`

---

**Happy Model Building with TensorFlow! 🚀**
