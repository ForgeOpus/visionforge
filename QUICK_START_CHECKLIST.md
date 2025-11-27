# VisionForge - Quick Start Checklist

Use this checklist to get VisionForge deployment-ready in the shortest time possible.

---

## ✅ Phase 1: Fix Critical Missing Pieces (2-4 hours)

### 1.1 Copy Missing Frontend Configs
```bash
# Copy Tailwind config
cp project/frontend/tailwind.config.js frontend/
cp project/frontend/postcss.config.js frontend/ 2>/dev/null || echo "No postcss config"

# Copy public assets
mkdir -p frontend/public
cp project/frontend/public/vision_logo.png frontend/public/ 2>/dev/null || echo "⚠️  Logo not found"
cp project/frontend/public/favicon.ico frontend/public/ 2>/dev/null || echo "No favicon"
```

**Status:** ⬜ Not Started

---

### 1.2 Create Missing API Client for Frontend
Create `frontend/src/lib/api.ts`:

```typescript
/**
 * API client for local FastAPI server
 * Wraps LocalInferenceClient with additional utilities
 */

import { localClient } from './inference'

export const api = {
  // Health check
  async healthCheck() {
    return localClient.healthCheck()
  },

  // Chat with AI
  async chat(message: string, history: any[] = [], workflowState?: any) {
    return localClient.chat({
      message,
      history,
      workflowState,
    })
  },

  // Validate architecture
  async validateModel(nodes: any[], edges: any[]) {
    return localClient.validateModel({ nodes, edges })
  },

  // Export model
  async exportModel(nodes: any[], edges: any[], format: 'pytorch' | 'tensorflow', projectName: string) {
    return localClient.exportModel({ nodes, edges, format, projectName })
  },

  // Get node definitions
  async getNodeDefinitions(framework: 'pytorch' | 'tensorflow' = 'pytorch') {
    return localClient.getNodeDefinitions(framework)
  },
}

export default api
```

**Status:** ⬜ Not Started

---

### 1.3 Copy Missing Utilities
```bash
# Copy utility files needed by frontend
cp project/frontend/src/lib/exportImport.ts frontend/src/lib/
cp project/frontend/src/lib/localStorageService.ts frontend/src/lib/
cp project/frontend/src/lib/apiUtils.ts frontend/src/lib/
```

**Status:** ⬜ Not Started

---

### 1.4 Fix packages/core Component Imports

**Problem:** Components copied from `project/frontend/src/components/` have imports like:
```typescript
import { Button } from '@/components/ui/button'  // ❌ Won't work in package
```

**Solution:** Update to relative imports:
```typescript
import { Button } from './ui/button'  // ✅ Works
```

**Quick Fix Script:**
```bash
# Find all files with @/ imports in packages/core
cd packages/core/src
find . -name "*.tsx" -o -name "*.ts" | xargs grep -l "@/" > /tmp/files-to-fix.txt

# Manual fix each file OR use sed (careful!)
# sed -i "s|@/components/ui/|./ui/|g" components/*.tsx
```

**Status:** ⬜ Not Started

---

### 1.5 Copy Missing Hook to packages/core
```bash
# Copy useNodeSpecs hook
mkdir -p packages/core/src/hooks
cp project/frontend/src/lib/useNodeSpecs.ts packages/core/src/hooks/
```

Update `packages/core/src/hooks/index.ts`:
```typescript
export * from './use-mobile'
export * from './useNodeSpecs'
```

**Status:** ⬜ Not Started

---

## ✅ Phase 2: Install & Test (1-2 hours)

### 2.1 Install Dependencies
```bash
# Install all workspace dependencies
npm install

# If errors, check each workspace:
cd packages/core && npm install
cd ../frontend && npm install
```

**Expected Issues:**
- Missing peer dependencies
- Version conflicts
- Type errors

**Status:** ⬜ Not Started

---

### 2.2 Type Check Everything
```bash
# Check packages/core
cd packages/core
npm run type-check

# Check frontend
cd ../frontend
npm run type-check
```

**Fix common errors:**
1. Missing type definitions → `npm install @types/...`
2. Import errors → Fix paths
3. Missing exports → Add to index.ts

**Status:** ⬜ Not Started

---

### 2.3 Build Frontend
```bash
cd frontend
npm run build
```

**Success criteria:**
- ✅ Build completes without errors
- ✅ Output in `python/vision_forge/web/`
- ✅ `index.html` and `assets/` directory created

**Status:** ⬜ Not Started

---

### 2.4 Test Python Server
```bash
cd python
pip install -e ".[dev,ai]"

# Create .env
cp .env.example .env
# Add your API keys (optional)

# Run server
python -m vision_forge.server
```

**Success criteria:**
- ✅ Server starts on http://localhost:8000
- ✅ Health endpoint responds: `curl http://localhost:8000/api/health`
- ✅ Frontend accessible at http://localhost:8000

**Status:** ⬜ Not Started

---

### 2.5 Test CLI Commands
```bash
# Test init
vision-forge init

# Test start
vision-forge start --port 8001
```

**Success criteria:**
- ✅ CLI commands work
- ✅ Server starts
- ✅ Browser opens automatically

**Status:** ⬜ Not Started

---

## ✅ Phase 3: Cleanup (1 hour)

### 3.1 Remove Build Artifacts
```bash
# Run safe cleanup
rm -rf project/frontend/dist
rm -rf project/frontend/node_modules
rm -rf project/frontend_build
rm -rf project/staticfiles
find project/ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find project/ -name "*.pyc" -delete

# Check space saved
du -sh project/
```

**Status:** ⬜ Not Started

---

### 3.2 Mark project/ as Deprecated
```bash
cat > project/README_DEPRECATED.md << 'EOF'
# ⚠️ DEPRECATED

This directory is deprecated. Use the new structure:
- `packages/core/` - Shared components
- `frontend/` - Local desktop app
- `python/` - Python package

This code will be moved to private `visionforge-web` repository.
EOF
```

**Status:** ⬜ Not Started

---

### 3.3 Commit Cleanup
```bash
git add -A
git commit -m "Complete local version implementation and cleanup build artifacts"
git push
```

**Status:** ⬜ Not Started

---

## ✅ Phase 4: Documentation (30 min)

### 4.1 Update Root README.md
Add quick start instructions for new structure:

```markdown
## Quick Start - Local Desktop Version

### Installation
\`\`\`bash
pip install vision-forge
\`\`\`

### Setup
\`\`\`bash
vision-forge init
# Optionally add API keys to .env
\`\`\`

### Run
\`\`\`bash
vision-forge start
\`\`\`

## Development

See [ARCHITECTURE.md](ARCHITECTURE.md) for the new dual-version structure.
```

**Status:** ⬜ Not Started

---

### 4.2 Create CONTRIBUTING.md for New Structure
```bash
# Copy from template and adapt for new structure
```

**Status:** ⬜ Not Started

---

## 📊 Progress Tracker

| Phase | Task | Status | Time Est. |
|-------|------|--------|-----------|
| 1.1 | Copy configs | ⬜ | 10 min |
| 1.2 | Create API client | ⬜ | 30 min |
| 1.3 | Copy utilities | ⬜ | 10 min |
| 1.4 | Fix imports | ⬜ | 60 min |
| 1.5 | Copy hooks | ⬜ | 10 min |
| 2.1 | Install deps | ⬜ | 20 min |
| 2.2 | Type check | ⬜ | 30 min |
| 2.3 | Build frontend | ⬜ | 15 min |
| 2.4 | Test Python | ⬜ | 15 min |
| 2.5 | Test CLI | ⬜ | 10 min |
| 3.1 | Cleanup | ⬜ | 10 min |
| 3.2 | Mark deprecated | ⬜ | 5 min |
| 3.3 | Commit | ⬜ | 5 min |
| 4.1 | Update README | ⬜ | 15 min |
| 4.2 | CONTRIBUTING | ⬜ | 15 min |

**Total Estimated Time: 4-5 hours**

---

## 🚨 Common Issues & Solutions

### Issue 1: "Cannot find module '@visionforge/core'"
**Solution:**
```bash
rm -rf node_modules package-lock.json
npm install
```

### Issue 2: TypeScript errors in packages/core
**Solution:**
1. Check import paths are relative, not using `@/`
2. Ensure all types are exported
3. Run `npm run type-check` in packages/core first

### Issue 3: Frontend build fails
**Solution:**
1. Check all dependencies installed
2. Ensure tailwind.config.js exists
3. Check vite.config.ts has correct paths

### Issue 4: Python server can't find frontend
**Solution:**
1. Ensure `npm run build` completed successfully
2. Check `python/vision_forge/web/` directory exists
3. Check `index.html` is in web/ directory

### Issue 5: CLI commands not found
**Solution:**
```bash
cd python
pip install -e .
# Verify: which vision-forge
```

---

## ✅ Definition of Done

The implementation is complete when:

- [ ] `npm install` succeeds without errors
- [ ] `npm run type-check` passes in all workspaces
- [ ] `cd frontend && npm run build` succeeds
- [ ] `python -m vision_forge.server` starts successfully
- [ ] Frontend loads at http://localhost:8000
- [ ] `vision-forge init` and `vision-forge start` work
- [ ] Build artifacts removed from project/
- [ ] Documentation updated
- [ ] Changes committed and pushed

---

## 🎯 Next Steps After Completion

1. Create GitHub release tag `v0.1.0-alpha`
2. Test installation from scratch: `pip install vision-forge`
3. Create private `visionforge-web` repository
4. Migrate Django backend to web repo
5. Publish `@visionforge/core` to npm
6. Publish `vision-forge` to PyPI

---

**Ready to start? Begin with Phase 1.1!**
