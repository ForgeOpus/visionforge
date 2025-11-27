# VisionForge - Local Desktop Version

Build neural networks visually and export production-ready code.

## Installation

```bash
pip install vision-forge
```

### Optional: AI Features

To enable AI-powered assistance (chatbot, suggestions):

```bash
pip install vision-forge[ai]
```

## Quick Start

### 1. Initialize Configuration

```bash
vision-forge init
```

This creates a `.env` file where you can optionally add your API keys for AI features:

```env
# Optional: For AI assistant features
GEMINI_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 2. Start VisionForge

```bash
vision-forge start
```

The application will open at `http://localhost:8000`

## Features

- 🎨 **Drag-and-drop interface** - Build CNNs, LSTMs, ResNets visually
- ⚡ **Automatic shape inference** - No manual tensor dimension tracking
- 🔄 **Multi-framework export** - PyTorch or TensorFlow
- 🤖 **AI assistant** (optional) - Natural language model modifications
- ✅ **Real-time validation** - Catch errors before export
- 🔒 **Privacy-first** - All data stays on your machine

## Usage

### Without AI Features

You can use VisionForge completely offline:
- Build architectures visually
- Validate models
- Export code to PyTorch/TensorFlow

No API keys needed!

### With AI Features

Add API keys to `.env` for:
- AI-powered chatbot
- Architecture suggestions
- Natural language modifications

## CLI Commands

```bash
# Initialize configuration
vision-forge init

# Start the application (default port 8000)
vision-forge start

# Start on custom port
vision-forge start --port 3000

# Start with specific host
vision-forge start --host 0.0.0.0

# Show version
vision-forge --version

# Show help
vision-forge --help
```

## Configuration

VisionForge looks for `.env` in the current directory. Example:

```env
# Server settings
HOST=127.0.0.1
PORT=8000

# Optional: AI features
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## Architecture

VisionForge runs a local web server (FastAPI) and serves a React frontend. All processing happens on your machine.

```
┌─────────────────────────────────────┐
│  Browser (React Frontend)           │
│  http://localhost:8000               │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│  FastAPI Server (Python)            │
│  - Code generation                   │
│  - Validation                        │
│  - Shape inference                   │
│  - AI services (optional)            │
└─────────────────────────────────────┘
```

## Development

To contribute or modify VisionForge:

```bash
# Clone the repository
git clone https://github.com/ForgeOpus/visionforge.git
cd visionforge/python

# Install in development mode
pip install -e ".[dev,ai]"

# Run tests
pytest

# Format code
black vision_forge/
ruff check vision_forge/
```

## License

BSD-3-Clause - See LICENSE in repository root

## Support

- Issues: https://github.com/ForgeOpus/visionforge/issues
- Discussions: https://github.com/ForgeOpus/visionforge/discussions
