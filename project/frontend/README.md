# Visual AI Model Builder

A browser-based visual interface for designing neural network architectures through intuitive drag-and-drop interactions, with automatic dimension inference, real-time validation, and PyTorch code generation.

## Features

### 🎨 Visual Architecture Design
- Drag-and-drop neural network blocks onto an infinite canvas
- Connect blocks with animated edges showing data flow
- Real-time visual feedback on tensor shapes and dimensions
- Color-coded blocks by category (Input, Basic, Advanced, Merge)

### 🧮 Automatic Dimension Inference
- Tensor shapes automatically computed as blocks connect
- Propagates dimensions through the entire graph
- Instant feedback on shape compatibility
- No manual dimension calculation needed

### ✅ Real-Time Validation
- Invalid connections prevented with helpful error messages
- Shape compatibility checked before connection creation
- Required parameters highlighted
- Architecture validation before code export
- Visual error indicators (red exclamation badges) on nodes with errors

### 💾 Project Management
- Save and load multiple projects
- Projects persist in browser storage
- Track architecture changes over time
- Framework selection (PyTorch/TensorFlow)

### 🚀 Export & Import
- **PyTorch Code**: Complete model class, training script, and configuration file
- **JSON Architecture**: Export/import entire architectures in a secure, readable format
- Copy code to clipboard or download as files
- Import previously exported architectures to continue working

## Available Blocks

### Input Layers
- **Input**: Define model input specifications with shape configuration

### Basic Layers
- **Linear**: Fully connected layer
- **Conv2D**: 2D convolutional layer
- **Dropout**: Dropout regularization
- **BatchNorm**: Batch normalization
- **ReLU**: Rectified Linear Unit activation
- **Softmax**: Softmax activation
- **Flatten**: Flatten tensor to 2D
- **MaxPool2D**: 2D max pooling

### Advanced Layers
- **Multi-Head Attention**: Self-attention mechanism

### Merge/Split
- **Concatenate**: Combine multiple tensors

## Getting Started

### 1. Create a New Project
Click "New Project" in the header and provide:
- Project name
- Optional description
- Framework (PyTorch or TensorFlow)

### 2. Build Your Architecture
1. Drag an **Input** block from the palette onto the canvas
2. Configure the input shape (batch size, channels, height, width)
3. Drag additional blocks and connect them by clicking handles
4. Configure each block's parameters in the right panel when selected
5. Watch dimensions automatically update as you connect blocks

### 3. Validate Your Design
- Invalid connections are prevented automatically
- Red exclamation badges appear on nodes with errors
- Orange text shows missing required parameters
- Click "Validate" button to check entire architecture

### 4. Export Your Architecture
1. Click "Export" dropdown in the header
2. Choose export format:
   - **PyTorch Code**: View generated model.py, train.py, and config.py
   - **JSON Architecture**: Download architecture as a portable JSON file
3. Copy code to clipboard or download files
4. Use the code in your own projects!

### 5. Save & Share Your Work
- **Save**: Click "Save" to persist your project in browser storage
- **Import**: Click "Import" to load a previously exported JSON architecture
- **Load**: Access saved projects from the project dropdown menu

## Example Architectures

### Simple CNN Classifier
```
Input → Conv2D → ReLU → MaxPool2D → Flatten → Linear
```

### Deep MLP
```
Input → Flatten → Linear → ReLU → Dropout → Linear
```

### Custom Architecture
Combine blocks in any valid configuration to create your unique model!

## Keyboard Shortcuts
- **Delete**: Remove selected block
- **Escape**: Deselect block
- **Click canvas**: Deselect current block

## Tips & Tricks

### Shape Compatibility
- **Linear layers** require 2D input - add Flatten before Linear
- **Conv2D layers** require 4D input [batch, channels, height, width]
- Use **Flatten** to convert from convolutional to fully-connected layers

### Common Patterns
1. **Image Classification**: Input → Conv2D → ReLU → MaxPool → Flatten → Linear
2. **Regularization**: Add Dropout after activation layers
3. **Normalization**: Add BatchNorm after convolutional layers

### Best Practices
- Start with an Input block to define your data shape
- Configure all required parameters (marked with *)
- Validate before exporting to catch errors early
- Save frequently to avoid losing work

## Technical Details

### Built With
- **React 19** - UI framework
- **ReactFlow** - Canvas and node-based interface
- **Zustand** - State management
- **Tailwind CSS** - Styling
- **shadcn/ui** - Component library
- **TypeScript** - Type safety

### Browser Storage
Projects are saved to browser localStorage using the Spark KV API. Your data stays on your device and never leaves your browser.

### Code Generation
The generated PyTorch code is production-ready and follows best practices:
- Proper `nn.Module` structure
- Configurable training loop
- Device selection (CPU/GPU)
- Checkpoint saving
- Standard optimizer and loss configurations

## Limitations

This is a frontend-only application that runs entirely in your browser:
- No server-side processing
- Projects saved locally (not synced across devices)
- Code generation produces PyTorch templates (you provide the dataset)
- Dimension inference may not handle all edge cases

## Sample Projects

The app comes with 3 example projects to get you started:
1. **Simple CNN Classifier** - Basic image classification architecture
2. **Attention Network** - Sequence processing with attention
3. **Deep MLP** - Multi-layer perceptron with regularization

Load these from the "Load" menu to explore and learn!

## Next Steps

- Add BatchNorm layers between Conv2D and activations for training stability
- Create ResNet-style architectures with skip connections using Concatenate blocks
- Export and test the generated PyTorch code with your own dataset
- Experiment with different layer combinations and configurations

---

Built with ❤️ using the Spark platform
