# VisionForge Frontend - Local Version

This is the frontend for the local desktop version of VisionForge.

## Key Differences from Web Version

- Uses `LocalInferenceClient` that communicates with local Python server
- No API key management in frontend (keys in `.env` on server side)
- Builds directly into `python/vision_forge/web` for packaging

## Development

```bash
# Install dependencies (from root)
npm install

# Start dev server (with hot reload)
npm run dev

# Build for production (outputs to python package)
npm run build

# Build locally (outputs to dist/)
npm run build:local
```

## Architecture

```
Frontend (React) → LocalInferenceClient → FastAPI Server → AI Services
                                                          → Code Generation
```

The frontend never sees API keys - they're read from `.env` by the Python server.

## Dependencies

This frontend depends on:
- `@visionforge/core` - Shared UI components, hooks, and types
- React, TypeScript, Vite
- Tailwind CSS for styling
- XYFlow for canvas

Changes to `@visionforge/core` are automatically reflected due to workspace linking.
