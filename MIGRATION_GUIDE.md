# Migration Guide - New Dual-Version Architecture

This guide explains the new structure and how to work with it.

## What Changed?

VisionForge now has a **hybrid structure** supporting two versions:

1. **Local Desktop** (`frontend/` + `python/`) - For PyPI distribution
2. **Web Cloud** (private repo) - For hosted deployment

The old `project/` structure is kept temporarily for reference and web version development.

## New Directory Structure

```
✨ NEW:
├── packages/core/          # Shared UI components (npm package)
├── frontend/               # Local version frontend
├── python/                 # Python package for PyPI
└── package.json            # Workspace root

📦 KEPT (for now):
└── project/                # Legacy structure for web version reference
```

## Quick Start with New Structure

### For Local Development

```bash
# 1. Install all dependencies
npm install

# 2. Start Python backend
cd python
pip install -e ".[dev,ai]"
python -m vision_forge.server

# 3. Start frontend (in new terminal)
cd frontend
npm run dev
```

### For Python Package Testing

```bash
# Build frontend into Python package
cd frontend
npm run build

# Install and run
cd python
pip install -e ".[ai]"
vision-forge init
vision-forge start
```

## File Location Changes

### Where Files Moved To

| Old Location | New Location | Purpose |
|-------------|--------------|---------|
| `project/frontend/src/lib/types.ts` | `packages/core/src/lib/types.ts` | Shared types |
| `project/frontend/src/lib/store.ts` | `packages/core/src/lib/store.ts` | Shared store |
| `project/frontend/src/lib/nodes/` | `packages/core/src/lib/nodes/` | Node definitions |
| `project/frontend/src/lib/validation/` | `packages/core/src/lib/validation/` | Validation |
| `project/frontend/src/components/ui/` | `packages/core/src/components/ui/` | UI components |
| `project/frontend/src/components/Canvas.tsx` | `packages/core/src/components/Canvas.tsx` | Canvas component |
| `project/block_manager/services/` | `python/vision_forge/services/` | Backend services |

**Note**: Original files in `project/` are kept for web version development.

## Working with the New Structure

### Editing Shared Components

```bash
# Components in packages/core are automatically linked
cd packages/core/src/components
# Edit Canvas.tsx, BlockPalette.tsx, etc.

# Changes are immediately reflected in frontend/ due to workspace link
cd ../../frontend
npm run dev  # Will use latest from packages/core
```

### Editing Local-Specific Code

```bash
cd frontend/src
# Edit App.tsx, local-client.ts, etc.
# These files are specific to local version
```

### Editing Python Backend

```bash
cd python/vision_forge
# Edit server.py, cli.py, services/, etc.
python -m vision_forge.server  # Test changes
```

## Important: What NOT to Delete

### Keep `project/` Directory

**Why?** It contains:
- Django backend (will move to private web repo later)
- Original AI services (copied to `python/`, but keep original)
- Reference frontend implementation

**When to delete?** After web version is fully migrated to private repository.

## API Key Changes

### Before (Monolithic)
```typescript
// API key in frontend, sent with requests
const apiKey = sessionStorage.getItem('api_key')
fetch('/api/chat', {
  headers: { 'X-Gemini-Api-Key': apiKey }
})
```

### After (Local Version)
```typescript
// No API key in frontend
fetch('/api/chat', {
  // Server reads key from .env
})
```

### After (Web Version - Private Repo)
```typescript
// Still uses sessionStorage (web-specific)
const apiKey = sessionStorage.getItem('api_key')
fetch('/api/chat', {
  headers: { 'X-Gemini-Api-Key': apiKey }
})
```

## Common Tasks

### Add a New Shared Component

```bash
# 1. Create in packages/core
cd packages/core/src/components
# Create NewComponent.tsx

# 2. Export it
# Add to packages/core/src/components/index.ts

# 3. Use in local frontend
cd frontend/src
# Import from @visionforge/core
```

### Add a Local-Only Feature

```bash
# Add to frontend/src (not packages/core)
cd frontend/src/components
# Create LocalFeature.tsx

# This won't be shared with web version
```

### Add a Python Backend Feature

```bash
cd python/vision_forge
# Edit server.py or add new service

# Test immediately
python -m vision_forge.server
```

### Build for Distribution

```bash
# Local desktop version
cd frontend
npm run build  # → python/vision_forge/web/

cd python
python -m build  # → dist/*.whl

# Core package (when ready to publish)
cd packages/core
npm publish
```

## Testing Your Changes

### Test Local Version End-to-End

```bash
# Terminal 1: Python server
cd python
python -m vision_forge.server

# Terminal 2: Frontend dev server
cd frontend
npm run dev

# Open http://localhost:5173
```

### Test Python Package Installation

```bash
# Build everything
cd frontend && npm run build
cd ../python && pip install -e ".[ai]"

# Run CLI
vision-forge init
vision-forge start
```

## Troubleshooting

### "Module not found: @visionforge/core"

```bash
# Reinstall workspace dependencies
rm -rf node_modules
npm install
```

### "Cannot find module 'vision_forge'"

```bash
# Install Python package
cd python
pip install -e ".[ai]"
```

### Frontend build fails

```bash
# Check TypeScript
cd packages/core
npm run type-check

cd ../frontend
npm run type-check
```

### Port already in use

```bash
# Change port
vision-forge start --port 3000

# Or in frontend
npm run dev -- --port 5174
```

## Next Steps

1. ✅ New structure is set up
2. 🚧 Test the local version workflow
3. 🚧 Create private web repository
4. 🚧 Migrate Django backend to private repo
5. 🚧 Publish `@visionforge/core` to npm
6. 🚧 Publish `vision-forge` to PyPI

## Need Help?

- See `ARCHITECTURE.md` for detailed structure docs
- See `python/README.md` for Python package usage
- See `packages/core/README.md` for core package details
- Open an issue if you're stuck!
