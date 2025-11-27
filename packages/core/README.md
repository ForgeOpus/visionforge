# @visionforge/core

Core components and shared logic for VisionForge - the visual neural network builder.

## Overview

This package contains:
- **Components**: Reusable React components (Canvas, BlockPalette, UI components)
- **Hooks**: Custom React hooks
- **Types**: Shared TypeScript types and interfaces
- **Node Definitions**: Neural network layer specifications
- **Validation**: Architecture validation logic
- **Inference Interface**: Abstract base classes for AI inference

## Installation

```bash
npm install @visionforge/core
```

## Usage

```typescript
import { Canvas, BlockPalette } from '@visionforge/core/components'
import { BaseInferenceClient } from '@visionforge/core/inference'
import { useNodeSpecs } from '@visionforge/core/hooks'
```

## For Developers

This package is part of the VisionForge monorepo. To work on it:

```bash
# Install dependencies
npm install

# Type check
npm run type-check

# Lint
npm run lint
```

## License

BSD-3-Clause - see LICENSE in repository root
