# VisionForge Documentation

This directory contains the complete user documentation for VisionForge, built with MkDocs.

## 📚 Documentation Overview

The documentation provides comprehensive guides for building neural networks visually using VisionForge's drag-and-drop interface.

### 🎯 Key Sections

- **Getting Started** - Installation and quick start guide
- **Architecture Design** - Creating diagrams and connection rules
- **Layer Reference** - Complete guide to all available layers
- **Examples** - Step-by-step tutorials for common architectures
- **Code Generation** - Export to PyTorch and TensorFlow
- **API Reference** - Backend and frontend API documentation
- **Advanced Topics** - Group blocks, AI assistant, sharing
- **Troubleshooting** - Common issues and solutions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install mkdocs mkdocs-material mkdocs-mermaid2-plugin mkdocs-video-plugin mkdocs-glightbox
```

### 2. Serve Documentation

```bash
mkdocs serve
```

The documentation will be available at `http://127.0.0.1:8000`

### 3. Build for Production

```bash
mkdocs build
```

Built files will be in the `site/` directory.

## 📖 Documentation Structure

```
docs/
├── index.md                           # Main landing page
├── getting-started/
│   ├── installation.md               # Setup instructions
│   ├── quickstart.md                 # First neural network
│   └── interface.md                  # UI overview
├── architecture/
│   ├── creating-diagrams.md          # Visual building guide
│   ├── connection-rules.md           # Layer compatibility
│   ├── shape-inference.md            # Tensor dimensions
│   └── validation.md                 # Error checking
├── layers/
│   ├── input.md                      # Input layer types
│   ├── core.md                       # Basic neural network layers
│   ├── activation.md                 # Activation functions
│   ├── pooling.md                    # Pooling operations
│   ├── merge.md                      # Combining paths
│   └── advanced.md                   # Specialized layers
├── examples/
│   ├── simple-cnn.md                 # Basic CNN tutorial
│   ├── resnet.md                     # Skip connections
│   ├── lstm.md                       # Sequence modeling
│   └── group-blocks.md               # Custom components
├── codegen/
│   ├── pytorch.md                    # PyTorch export
│   ├── tensorflow.md                 # TensorFlow export
│   └── custom-templates.md           # Custom code generation
├── api/
│   ├── rest-api.md                   # Backend API
│   └── node-definitions.md           # Layer specifications
├── advanced/
│   ├── group-blocks.md               # Reusable components
│   ├── ai-assistant.md               # Natural language help
│   └── sharing.md                    # Project collaboration
└── troubleshooting/
    ├── common-issues.md              # Frequently asked questions
    ├── validation-errors.md          # Architecture validation
    └── performance.md                # Optimization tips
```

## 🎨 Features

### Interactive Elements
- **Mermaid diagrams** for architecture visualization
- **Code highlighting** for generated examples
- **Responsive design** for all devices
- **Search functionality** for quick navigation

### Navigation
- **Tabbed navigation** by topic
- **Expandable sections** for detailed content
- **Breadcrumbs** for location tracking
- **Related links** for guided learning

### Visual Aids
- **Color-coded layers** by category
- **Connection diagrams** showing valid paths
- **Shape progression tables** for tensor tracking
- **Validation indicators** for error checking

## 🔧 Configuration

The documentation uses the Material theme with these features:

- **Dark/light mode** toggle
- **Code syntax highlighting**
- **Mermaid diagram support**
- **Responsive design**
- **Search functionality**
- **Social links**

## 📝 Contributing

When updating documentation:

1. **Follow the existing style** and structure
2. **Use Mermaid diagrams** for visual explanations
3. **Include code examples** with proper highlighting
4. **Add cross-references** to related topics
5. **Test all links** and examples

### Style Guidelines

- Use **clear headings** and subheadings
- Include **emoji** for visual hierarchy
- Provide **step-by-step** instructions
- Add **validation checklists** where appropriate
- Include **common pitfalls** and solutions

## 🚀 Deployment

### GitHub Pages
```bash
# Install gh-pages
pip install gh-pages

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### Custom Domain
Update `mkdocs.yml` with your domain:
```yaml
site_url: https://your-domain.com/docs
```

## 📊 Analytics

Add Google Analytics by updating `mkdocs.yml`:
```yaml
extra:
  analytics:
    provider: google
    property: G-XXXXXXXXXX
```

## 🔗 External Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material Theme](https://squidfunk.github.io/mkdocs-material/)
- [Mermaid Diagram Syntax](https://mermaid-js.github.io/)

---

**For VisionForge usage**, see the [main documentation](docs/index.md)
