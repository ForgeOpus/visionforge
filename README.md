<div align="center">
  <img src="project/frontend/public/vision_logo.png" alt="VisionForge Logo" width="120" height="120">

  # VisionForge

  **Design Neural Networks Without Writing Code**

  Build production-ready AI models visually. Export clean PyTorch or TensorFlow code in minutes.

  [![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Built with Spark](https://img.shields.io/badge/Built%20with-Spark-orange)](https://github.com/github/spark)

</div>

---

## ✨ Features

### 🎨 **Intuitive Visual Builder**
Design complex neural network architectures with a drag-and-drop canvas. No coding required.

### 🔄 **Automatic Shape Inference**
Tensor dimensions calculated automatically as you build. No manual dimension tracking needed.

### ✅ **Intelligent Validation**
Real-time architecture validation catches errors instantly with helpful suggestions.

### 💾 **Save & Share Projects**
Export architectures as JSON. Import pre-built models or share with your team effortlessly.

### ⚡ **Multi-Framework Support**
Export to PyTorch or TensorFlow with a single click. Maintain flexibility in your ML workflow.

### 📝 **Clean, Production Code**
Generate well-structured, documented code with type hints and best practices built-in.

### 🔀 **Complex Architectures**
Build multi-branch models with skip connections, residual blocks, and merge operations (ResNet, U-Net, and beyond).

### 🛠️ **Custom Layer Support**
Extend with your own custom implementations. Full flexibility for research and production needs.

### 🤖 **AI-Powered Chatbot**
Intelligent assistant powered by Google Gemini. Ask questions or let AI modify your workflow in real-time.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### 1. Clone & Setup Backend

```bash
# Clone repository
git clone https://github.com/devgunnu/visionforge.git
cd visionforge/project

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Start Django server
python manage.py runserver
```

Backend runs on: `http://localhost:8000`

### 2. Setup Frontend

```bash
cd project/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend runs on: `http://localhost:5173`

### 3. Start Building!

Open `http://localhost:5173` in your browser and start designing your first neural network.

---

## 🎯 How It Works

1. **Drag & Drop** - Add blocks from the sidebar (Input, Conv2D, Linear, etc.)
2. **Connect** - Draw connections between blocks to define data flow
3. **Configure** - Click blocks to set parameters (filters, kernel size, etc.)
4. **Validate** - Real-time validation ensures your architecture is correct
5. **Export** - Generate production-ready PyTorch or TensorFlow code

---

## 🏗️ Tech Stack

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS
- **Canvas**: ReactFlow (@xyflow/react)
- **State**: Zustand
- **Animations**: Framer Motion
- **Backend**: Django, Python
- **AI**: Google Gemini API
- **UI**: Radix UI, shadcn/ui

---

## 📦 Available Blocks

### Core Layers
- **Input/Output** - Define model inputs and outputs
- **Linear** - Fully connected layers
- **Conv1D/2D/3D** - Convolutional layers
- **Flatten** - Flatten tensors for FC layers

### Activation & Normalization
- **ReLU, Sigmoid, Tanh, Softmax** - Activation functions
- **BatchNorm2D** - Batch normalization
- **Dropout** - Regularization

### Pooling
- **MaxPool2D, AvgPool2D** - Pooling layers
- **AdaptiveAvgPool2D** - Adaptive pooling

### Recurrent & Attention
- **LSTM, GRU** - Recurrent layers
- **Embedding** - Embedding layers

### Merge & Split
- **Add, Concat** - Combine multiple inputs
- **Custom** - Define your own layers

---

## 💡 Example Architectures

### Simple CNN Classifier
```
Input → Conv2D → ReLU → MaxPool2D → Flatten → Linear → Output → Loss
```

### ResNet-style Block
```
Input → Conv2D → BatchNorm → ReLU → Conv2D → Add (skip) → ReLU → Output
```

### Sequence Model
```
Input → Embedding → LSTM → Linear → Output → Loss
```

---

## 📖 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Detailed setup instructions
- **[Chatbot Setup](docs/CHATBOT_SETUP.md)** - Configure AI assistant
- **[Export Format](docs/EXPORT_FORMAT.md)** - Architecture JSON structure
- **[Node Architecture](docs/NODE_DEFINITION_ARCHITECTURE.md)** - Backend node system

---

## 🎮 Keyboard Shortcuts

- `Ctrl+Z` - Undo
- `Ctrl+Y` - Redo
- `Delete` - Remove selected block

---

## 🐛 Troubleshooting

### Backend not running?
```bash
cd project
python manage.py runserver
```

### Frontend errors?
```bash
cd project/frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Database issues?
```bash
cd project
python manage.py migrate
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

Built with ❤️ using [GitHub Spark](https://github.com/github/spark)

---

<div align="center">

**Ready to build your first AI model?**

[Get Started](#-quick-start) • [View Docs](docs/) • [Report Bug](https://github.com/devgunnu/visionforge/issues)

</div>
