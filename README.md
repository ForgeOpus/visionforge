<div align="center">
  <img src="project/frontend/public/vision_logo.png" alt="VisionForge Logo" width="200">

  # VisionForge

  **Build Neural Networks Visually — Export Production Code**

  Design deep learning architectures with drag-and-drop. Export clean PyTorch or TensorFlow code instantly.

  [![BSD-3-Clause License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
  [![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev)

</div>

<br />

## ✨ What is VisionForge?

VisionForge is a **visual neural network builder** that lets you design complex deep learning architectures without writing code. Perfect for researchers, students, and ML engineers who want to rapidly prototype models.

- 🎨 **Drag-and-drop interface** — Build CNNs, LSTMs, ResNets visually
- ⚡ **Automatic shape inference** — No manual tensor dimension tracking
- 🔄 **Multi-framework export** — PyTorch or TensorFlow with one click
- 🤖 **AI-powered assistant** — Ask questions or modify your model with natural language
- ✅ **Real-time validation** — Catch architecture errors before export

<br />

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- [Google Gemini API key](https://aistudio.google.com/app/apikey) (for AI assistant)

### Installation

**1. Clone and setup backend**
```bash
git clone https://github.com/devgunnu/visionforge.git
cd visionforge/project

# Install Python dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Add your GEMINI_API_KEY to .env

# Initialize database
python manage.py migrate

# Start Django server
python manage.py runserver
```

Backend runs at `http://localhost:8000`

**2. Setup frontend**
```bash
cd project/frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend runs at `http://localhost:5173`

**3. Open your browser**
Navigate to `http://localhost:5173` and start building!

<br />

## 🎯 How It Works

<div align="center">

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  Drag & Drop │ →  │  Configure   │ →  │   Validate   │ →  │   Export    │
│    Blocks    │    │  Parameters  │    │ Architecture │    │    Code     │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

</div>

1. **Add layers** from the sidebar (Conv2D, LSTM, Dropout, etc.)
2. **Connect blocks** to define your model's data flow
3. **Set parameters** by clicking on any block
4. **Validate** your architecture with built-in checks
5. **Export** production-ready code for PyTorch or TensorFlow

<br />

## 📦 Available Layers

<table>
<tr>
<td width="50%">

**Core Layers**
- Input / Output
- Linear (Fully Connected)
- Conv1D / Conv2D / Conv3D
- Flatten, Reshape

**Activation & Regularization**
- ReLU, Sigmoid, Tanh, Softmax
- Dropout, BatchNorm
- Layer Normalization

</td>
<td width="50%">

**Pooling**
- MaxPool2D, AvgPool2D
- AdaptiveAvgPool2D

**Recurrent & Sequence**
- LSTM, GRU
- Embedding

**Operations**
- Add, Concatenate
- Custom layers

</td>
</tr>
</table>

<br />

## 💡 Example Architectures

**Simple CNN Classifier**
```
Input → Conv2D → ReLU → MaxPool2D → Flatten → Linear → Softmax → Loss
```

**ResNet-style Skip Connection**
```
Input → Conv2D → BatchNorm → ReLU ┐
                                   ├→ Add → ReLU → Output
       Input ────────────────────→┘
```

**LSTM Sequence Model**
```
Input → Embedding → LSTM → Dropout → Linear → Output → Loss
```

<br />

## 🛠️ Tech Stack

<table>
<tr>
<td><b>Frontend</b></td>
<td>React 19 • TypeScript • Vite • Tailwind CSS</td>
</tr>
<tr>
<td><b>Canvas</b></td>
<td>ReactFlow • Zustand • Framer Motion</td>
</tr>
<tr>
<td><b>Backend</b></td>
<td>Django • Python • SQLite</td>
</tr>
<tr>
<td><b>AI</b></td>
<td>Google Gemini API</td>
</tr>
<tr>
<td><b>UI</b></td>
<td>Radix UI • shadcn/ui</td>
</tr>
</table>

<br />

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Delete` | Remove selected block |

<br />

## 📖 Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [AI Chatbot Setup](docs/CHATBOT_SETUP.md)
- [Export Format Specification](docs/EXPORT_FORMAT.md)
- [Node Architecture](docs/NODE_DEFINITION_ARCHITECTURE.md)

<br />

## 🐛 Troubleshooting

**Backend not starting?**
```bash
cd project
python manage.py migrate
python manage.py runserver
```

**Frontend build errors?**
```bash
cd project/frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

**CORS issues?**
Make sure both servers are running (Django on 8000, Vite on 5173)

<br />

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit PRs.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

<br />

## 📄 License & Attribution

VisionForge is licensed under the **BSD 3-Clause License**.

```
Copyright (c) 2025, ForgeOpus

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the
LICENSE file are met.
```

See [LICENSE](LICENSE) for the full license text.

### Third-Party Software

VisionForge is built on top of many excellent open source projects. We are grateful to the open source community for their contributions.

**Key Dependencies:**
- **Frontend:** React, TypeScript, Vite, Tailwind CSS, Radix UI, XYFlow, Zustand, Framer Motion
- **Backend:** Django, Python, Google Generative AI, Anthropic
- **Total:** 350+ open source packages under permissive licenses (MIT, Apache-2.0, BSD, ISC)

For complete license information and attributions:
- 📋 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) - Comprehensive list of all dependencies and their licenses
- 📄 [NOTICE](NOTICE) - Required attribution notices for specific dependencies

### License Compliance

All dependencies use licenses compatible with BSD-3-Clause:
- ✅ **MIT, Apache-2.0, BSD, ISC** - Permissive licenses (majority of dependencies)
- ✅ **MPL-2.0** - Weak copyleft (LightningCSS) - properly attributed
- ✅ **LGPL** - Dynamically linked (Python libraries) - properly attributed

We are committed to open source license compliance. If you have concerns about license compliance, please [open an issue](https://github.com/ForgeOpus/visionforge/issues).

<br />

<div align="center">

---

**Ready to build AI models faster?**

[Get Started](#-quick-start) • [View Docs](docs/) • [Report Issues](https://github.com/ForgeOpus/visionforge/issues)

Made by [devgunnu](https://github.com/devgunnu) | Maintained by [ForgeOpus](https://github.com/ForgeOpus)

</div>
