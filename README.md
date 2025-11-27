<div align="center">
  <h1>🔮 VisionForge</h1>
  
  **Visual Neural Network Builder for PyTorch & TensorFlow**
  
  Design deep learning architectures with drag-and-drop blocks. Export production-ready code instantly.
  
  [![BSD-3-Clause License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
  [![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
  [![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev)
  
</div>

---

## ✨ Features

- 🎨 **Visual Architecture Design** — Drag-and-drop interface for building neural networks
- ⚡ **Automatic Shape Inference** — No manual tensor dimension tracking required
- 🔄 **Multi-Framework Export** — Generate PyTorch or TensorFlow code
- 🤖 **AI Assistant** — Chat interface powered by Gemini/Claude for model suggestions
- ✅ **Real-Time Validation** — Catch architecture errors before export
- 📦 **50+ Layer Types** — Conv, LSTM, Attention, Custom layers, and more

---

## 🚀 Quick Start

### Desktop Application (Recommended)

The desktop version runs entirely locally with a Python backend server.

**Prerequisites:**
- Python 3.8+
- Node.js 16+

**Installation:**

```bash
# Clone the repository
git clone https://github.com/ForgeOpus/visionforge.git
cd visionforge

# Install Python package
cd python
pip install -e ".[dev,ai]"

# Configure API keys (optional, for AI features)
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY or ANTHROPIC_API_KEY

# Start the backend server
python -m vision_forge.server
```

The server runs at `http://localhost:8000`

**Frontend:**

```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:5173`

Open `http://localhost:5173` in your browser and start building!

---

## 📦 Architecture

VisionForge is a monorepo with three main packages:

```
visionforge/
├── frontend/           # React + TypeScript desktop UI
├── packages/core/      # Shared React components & utilities
└── python/            # FastAPI server + code generation
```

### Package Overview

- **`frontend/`** — Desktop application with local API client
- **`packages/core/`** — Reusable UI components (Radix UI, validation, node definitions)
- **`python/`** — Python package with FastAPI server, code generation, AI integration

---

## 🎯 Building Your First Model

1. **Add Layers** — Drag blocks from the left sidebar (Input, Conv2D, Linear, etc.)
2. **Connect Blocks** — Draw connections to define data flow
3. **Configure** — Click any block to adjust parameters
4. **Validate** — Real-time checks ensure your architecture is valid
5. **Export** — Generate PyTorch or TensorFlow code with one click

### Example: Simple CNN Classifier

```
Input(28×28×1) → Conv2D(32) → ReLU → MaxPool2D → 
Flatten → Linear(10) → Softmax → Output
```

### Example: ResNet-Style Skip Connection

```
Input → Conv2D → BatchNorm → ReLU ┐
                                   ├→ Add → ReLU → Output
Input ─────────────────────────────┘
```

---

## 🧩 Available Layers

<table>
<tr>
<td width="33%">

**Core Layers**
- Input / Output
- Linear (Dense)
- Conv1D/2D/3D
- Flatten / Reshape
- Embedding

</td>
<td width="33%">

**Activations & Norm**
- ReLU, Sigmoid, Tanh
- Softmax, LogSoftmax
- BatchNorm, LayerNorm
- Dropout

</td>
<td width="33%">

**Advanced**
- LSTM, GRU
- Attention
- MaxPool, AvgPool
- Add, Concatenate
- Custom Layers

</td>
</tr>
</table>

---

## 🛠️ Development

### Build from Source

```bash
# Install all dependencies
npm install

# Build core package
npm run build:core

# Run frontend in dev mode
npm run dev

# Run Python tests
cd python && pytest
```

### Project Structure

```
frontend/
  src/
    components/     # React components (Canvas, BlockPalette, etc.)
    lib/           # API client, utilities, types
    
packages/core/
  src/
    components/ui/ # Radix UI wrappers
    lib/
      nodes/      # Node definitions & registry
      validation/ # Shape inference engine
      store/      # Zustand state management
      
python/
  vision_forge/
    server/       # FastAPI app
    codegen/      # PyTorch/TensorFlow code generation
    ai/          # Gemini/Claude integrations
```

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Delete` | Remove selected block |
| `Ctrl+S` | Save architecture |

---

## 🤖 AI Assistant

VisionForge includes an AI-powered chatbot that can:
- Suggest layer configurations
- Explain architectural patterns
- Recommend improvements
- Answer ML/DL questions

**Supported Providers:**
- Google Gemini (default)
- Anthropic Claude

Configure in `.env`:
```bash
GEMINI_API_KEY=your_key_here
# or
ANTHROPIC_API_KEY=your_key_here
```

---

## 📖 Documentation

- **Getting Started:** [Quick Start Guide](docs/QUICKSTART.md)
- **Architecture:** [System Design](ARCHITECTURE.md)
- **Nodes & Validation:** [Node Rules](docs/NODES_AND_RULES.md)
- **Contributing:** [Contribution Guide](CONTRIBUTING.md)

---

## 🧪 Testing

```bash
# Frontend tests
cd frontend && npm test

# Python tests
cd python && pytest

# Type checking
npm run type-check
```

---

## 🐛 Troubleshooting

**Backend won't start?**
```bash
cd python
pip install -e ".[dev,ai]"
python -m vision_forge.server
```

**Frontend build errors?**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

**Shape inference issues?**
Check [NODES_AND_RULES.md](docs/NODES_AND_RULES.md) for dimension requirements.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run linting and tests (`npm run lint && npm test`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push and open a Pull Request

---

## 📄 License

VisionForge is licensed under the **BSD 3-Clause License**.

```
Copyright (c) 2025, ForgeOpus
All rights reserved.
```

See [LICENSE](LICENSE) for full terms.

### Third-Party Software

VisionForge uses 350+ open source packages. See:
- [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) — Complete dependency list
- [NOTICE](NOTICE) — Attribution notices

All dependencies use BSD/MIT/Apache-2.0 compatible licenses.

---

## 🔗 Links

- **Repository:** [github.com/ForgeOpus/visionforge](https://github.com/ForgeOpus/visionforge)
- **Issues:** [Report bugs or request features](https://github.com/ForgeOpus/visionforge/issues)
- **Discussions:** [Community forum](https://github.com/ForgeOpus/visionforge/discussions)

---

<div align="center">

**Built with ❤️ by [ForgeOpus](https://github.com/ForgeOpus)**

[Get Started](#-quick-start) • [Documentation](docs/) • [Contributing](CONTRIBUTING.md)

</div>
