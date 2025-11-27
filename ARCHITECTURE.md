# VisionForge Architecture - Dual Version Setup

This document describes the new dual-version architecture for VisionForge.

## Overview

VisionForge now supports two versions:

1. **Local Desktop Version** (Public) - Python package with local AI support
2. **Web Cloud Version** (Private repo) - Hosted web app with API-based AI

Both versions share the same core UI components but differ in how they handle AI inference and API keys.

## Repository Structure

```
visionforge/ (Public Repository)
├── packages/
│   └── core/                        # 📦 @visionforge/core npm package
│       ├── src/
│       │   ├── components/          # Shared UI components
│       │   ├── hooks/               # Shared React hooks
│       │   ├── lib/
│       │   │   ├── inference/       # Abstract inference interfaces
│       │   │   ├── nodes/           # Node definitions
│       │   │   ├── validation/      # Validation engine
│       │   │   ├── types.ts
│       │   │   ├── store.ts
│       │   │   └── utils.ts
│       │   └── index.ts
│       └── package.json
│
├── frontend/                        # 🖥️ Local desktop frontend
│   ├── src/
│   │   ├── lib/inference/
│   │   │   └── local-client.ts      # Local inference implementation
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json                 # Depends on @visionforge/core
│   └── vite.config.ts              # Builds to python/vision_forge/web
│
├── python/                          # 🐍 Python package (PyPI)
│   ├── vision_forge/
│   │   ├── server.py                # FastAPI server
│   │   ├── cli.py                   # CLI commands
│   │   ├── services/                # AI services, code generation
│   │   └── web/                     # Built frontend (gitignored)
│   ├── pyproject.toml
│   └── README.md
│
├── project/                         # ⚠️ Legacy structure (keep for now)
│   ├── backend/                     # Django backend (for web version)
│   ├── block_manager/               # Services (copied to python/)
│   └── frontend/                    # Old frontend
│
└── package.json                     # Workspace root
```

## Key Components

### 1. `packages/core` - Shared UI Library

**Purpose**: Contains all shared code between local and web versions.

**Exports**:
- Components: `Canvas`, `BlockPalette`, `ChatBot`, UI components
- Hooks: `useNodeSpecs`, etc.
- Types: All TypeScript types and interfaces
- Validation: Architecture validation logic
- Inference: Abstract `BaseInferenceClient` class

**Publishing**: Will be published to npm as `@visionforge/core`

**Key File**: `src/lib/inference/base.ts`
```typescript
export abstract class BaseInferenceClient {
  abstract chat(options: ChatOptions): Promise<ChatResponse>
  abstract validateModel(options: ValidationOptions): Promise<ValidationResponse>
  abstract exportModel(options: ExportOptions): Promise<ExportResponse>
}
```

### 2. `frontend/` - Local Desktop Frontend

**Purpose**: Frontend for the local Python package version.

**Key Features**:
- Uses `LocalInferenceClient` that talks to local FastAPI server
- No API key management (keys handled server-side from `.env`)
- Builds directly into `python/vision_forge/web/` for packaging

**Key File**: `src/lib/inference/local-client.ts`
```typescript
export class LocalInferenceClient extends BaseInferenceClient {
  // Talks to http://localhost:8000/api/*
  // No API keys in requests - server reads from .env
}
```

### 3. `python/` - Python Package

**Purpose**: Distributable Python package for local desktop use.

**Distribution**: PyPI as `vision-forge`

**Installation**:
```bash
pip install vision-forge
vision-forge init    # Creates .env
vision-forge start   # Launches server + opens browser
```

**Key Features**:
- FastAPI server on localhost
- Reads API keys from `.env` file (never exposed to frontend)
- Includes AI services (Gemini, Claude)
- Code generation (PyTorch, TensorFlow)
- Bundles built frontend in `web/` directory

**CLI Commands**:
- `vision-forge init` - Interactive setup, creates `.env`
- `vision-forge start` - Start server on localhost:8000
- `vision-forge start --port 3000` - Custom port
- `vision-forge --version` - Show version

### 4. Private Web Repository (Not in this repo)

**Structure**:
```
visionforge-web/ (Private Repository)
├── src/
│   ├── lib/inference/
│   │   └── api-client.ts           # Cloud API implementation
│   ├── components/
│   │   └── ApiKeyModal.tsx         # Web-only component
│   └── App.tsx
├── backend/                         # Django backend
└── package.json                     # Depends on @visionforge/core from npm
```

**Key Difference**: Uses `ApiInferenceClient` that sends API keys in request headers.

## API Key Handling

### Local Version (Public)
```
User's .env file:
  GEMINI_API_KEY=xxx
  ANTHROPIC_API_KEY=yyy
         ↓
  Python FastAPI Server
  (reads from environment)
         ↓
  Frontend (never sees keys)
```

**Security**: API keys never leave the local machine, never sent to frontend.

### Web Version (Private)
```
User enters key in modal
         ↓
  sessionStorage (browser)
         ↓
  Sent in X-Gemini-Api-Key header
         ↓
  Django backend API
```

**Security**: Keys stored in browser session, sent with each request.

## Workflow for Changes

### UI Component Changes

1. **Edit** in `packages/core/src/components/`
2. **Test locally** using workspace link:
   ```bash
   cd frontend
   npm run dev  # Uses workspace:* link to packages/core
   ```
3. **Publish** when ready:
   ```bash
   cd packages/core
   npm version patch
   npm publish
   ```
4. **Update web version**:
   ```bash
   cd visionforge-web
   npm update @visionforge/core
   ```

### Local Version Changes

**Frontend**:
- Edit `frontend/src/` (local-specific code)
- `npm run build` → outputs to `python/vision_forge/web/`

**Backend**:
- Edit `python/vision_forge/`
- Test with `python -m vision_forge.server`

**Publish**:
```bash
cd python
python -m build
twine upload dist/*
```

### Web Version Changes (in private repo)

- Edit `visionforge-web/src/` (web-specific code)
- Edit `visionforge-web/backend/` (Django backend)
- Deploy to hosting platform

## Development Setup

### Initial Setup

```bash
# Install all workspace dependencies
npm install

# Install Python package in dev mode
cd python
pip install -e ".[dev,ai]"
```

### Development Workflow

```bash
# Terminal 1: Frontend dev server
cd frontend
npm run dev
# Opens on localhost:5173, proxies API to localhost:8000

# Terminal 2: Python server
cd python
python -m vision_forge.server
# Runs on localhost:8000

# Terminal 3: Watch packages/core changes
cd packages/core
npm run type-check -- --watch
```

### Testing the Full Stack

1. Start Python server: `cd python && python -m vision_forge.server`
2. Start frontend: `cd frontend && npm run dev`
3. Open `http://localhost:5173`
4. Frontend talks to Python backend via proxy

## Build Process

### Local Version (for PyPI)

```bash
# Build frontend into Python package
cd frontend
npm run build
# Outputs to python/vision_forge/web/

# Build Python package
cd python
python -m build
# Creates dist/vision_forge-0.1.0-py3-none-any.whl

# Test installation
pip install dist/vision_forge-0.1.0-py3-none-any.whl
vision-forge start
```

### Core Package (for npm)

```bash
cd packages/core
npm run build
npm publish
```

## Migration Status

### ✅ Completed

- Core package structure created
- Base inference interfaces defined
- Shared components copied to `packages/core`
- Python package structure with FastAPI server
- CLI commands (`vision-forge init`, `vision-forge start`)
- Local frontend with `LocalInferenceClient`
- Workspace configuration

### 🚧 Pending

- Publish `@visionforge/core` to npm
- Test full local installation workflow
- Create private `visionforge-web` repository
- Implement `ApiInferenceClient` for web version
- Migrate Django backend to private repo
- CI/CD pipelines for both versions

### 📝 To Keep

- `project/` directory - Keep for now as reference for web version
- Existing `project/block_manager/` - Services copied to `python/`, keep original for web version

## Benefits

| Benefit | How Achieved |
|---------|-------------|
| **Shared UI** | `@visionforge/core` npm package |
| **No overwrites** | Separate `local-client.ts` vs `api-client.ts` |
| **Easy updates** | `npm update @visionforge/core` in web repo |
| **Dev speed** | Workspace linking for instant changes |
| **Security** | Local keys in `.env`, web keys in session |
| **One codebase** | Contributors edit one repo for UI |

## Questions?

See:
- `python/README.md` - Python package usage
- `packages/core/README.md` - Core package details
- `frontend/README.md` - Local frontend details
