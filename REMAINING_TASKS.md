# VisionForge - Remaining Tasks & Cleanup Plan

## 🔍 Current State Analysis

### What's Working ✅
- ✅ Core package structure created (`packages/core/`)
- ✅ Python package structure created (`python/`)
- ✅ Frontend structure created (`frontend/`)
- ✅ Base inference interfaces defined
- ✅ CLI commands implemented
- ✅ FastAPI server with .env support
- ✅ Workspace configuration

### What's NOT Yet Done ❌

#### 1. **Missing Core Implementations**
- ❌ `packages/core` components have import issues (importing from project paths)
- ❌ `frontend/src/App.tsx` references components that may not exist yet
- ❌ Missing `useNodeSpecs` hook implementation in `packages/core`
- ❌ Missing API utilities in `frontend/` (was in `project/frontend/src/lib/api.ts`)
- ❌ No tailwind config in `frontend/`
- ❌ No public assets (logo, etc.) in `frontend/public/`

#### 2. **Build & Testing Issues**
- ❌ `npm install` not run - dependencies not installed
- ❌ TypeScript compilation not tested
- ❌ Frontend build not tested
- ❌ Python package not tested
- ❌ End-to-end workflow not verified

#### 3. **Missing Glue Code**
- ❌ API client for validation/export endpoints not implemented in `frontend/`
- ❌ Store provider setup in `frontend/src/App.tsx`
- ❌ ReactFlow provider setup
- ❌ Theme provider setup (for Radix UI)

#### 4. **Documentation Gaps**
- ❌ No CONTRIBUTING.md for new structure
- ❌ No testing instructions
- ❌ No deployment guide

---

## 📋 Remaining Implementation Tasks

### Phase 1: Fix Core Package (Priority: HIGH)
1. Fix import paths in all copied components
2. Copy missing hooks (`useNodeSpecs.ts`)
3. Add proper exports to `packages/core/src/index.ts`
4. Create barrel exports for component directories
5. Fix TypeScript errors

### Phase 2: Complete Frontend Setup (Priority: HIGH)
1. Copy tailwind config from `project/frontend/`
2. Copy public assets (logo, favicon)
3. Implement API utilities for local server
4. Set up proper App.tsx with all providers
5. Create proper routing (if needed)
6. Add error boundaries

### Phase 3: Test & Validate (Priority: HIGH)
1. Run `npm install` and fix dependency issues
2. Test TypeScript compilation in all workspaces
3. Build frontend and verify output
4. Test Python server startup
5. Test CLI commands
6. End-to-end integration test

### Phase 4: Optimization (Priority: MEDIUM)
1. Remove unused dependencies
2. Optimize bundle size
3. Add build scripts
4. Add development scripts

---

## 🧹 Cleanup & Redundancy Removal Plan

### Directory Size Analysis
```
18M   project/          # 97% of total size - MOST REDUNDANT
460K  packages/         # New core package
50K   frontend/         # New frontend (minimal so far)
495K  python/           # New python package
```

### Redundant Files to Address

#### Category A: Duplicate Code (Can Remove After Migration)
```
project/frontend/src/lib/
├── store.ts              → COPIED to packages/core/src/lib/store.ts
├── types.ts              → COPIED to packages/core/src/lib/types.ts
├── utils.ts              → COPIED to packages/core/src/lib/utils.ts
├── nodes/                → COPIED to packages/core/src/lib/nodes/
└── validation/           → COPIED to packages/core/src/lib/validation/

project/frontend/src/components/
├── Canvas.tsx            → COPIED to packages/core/src/components/
├── BlockPalette.tsx      → COPIED to packages/core/src/components/
├── ChatBot.tsx           → COPIED to packages/core/src/components/
└── ui/                   → COPIED to packages/core/src/components/ui/

project/block_manager/services/
├── gemini_service.py     → COPIED to python/vision_forge/services/
├── claude_service.py     → COPIED to python/vision_forge/services/
├── inference.py          → COPIED to python/vision_forge/services/
└── ...                   → COPIED to python/vision_forge/services/
```

#### Category B: Web-Specific (Move to Private Repo)
```
project/frontend/src/lib/
├── apiKeyContext.tsx     → Move to visionforge-web (web only)
└── api.ts                → Adapt for visionforge-web (Django endpoints)

project/frontend/src/components/
└── ApiKeyModal.tsx       → Move to visionforge-web (web only)

project/backend/          → Move to visionforge-web (Django backend)
project/block_manager/    → Keep services copy in python/, move Django models to web
```

#### Category C: Legacy/Unused (Can Delete)
```
project/frontend/dist/              → Build artifacts (gitignored)
project/frontend/node_modules/      → Dependencies (gitignored)
project/frontend_build/             → Old build output
project/staticfiles/                → Django static files
project/backend/__pycache__/        → Python cache
project/backend/db.sqlite3          → Development database
```

---

## 🚀 Deployment-Ready Action Plan

### **Step 1: Complete Missing Implementations** (CRITICAL)

**Files to Create/Fix:**

1. **Copy missing utilities to frontend/**
```bash
# Need to create adapted versions for local client
frontend/src/lib/
├── api.ts              # Local API client (talks to FastAPI)
├── exportImport.ts     # Copy from project/frontend
└── localStorageService.ts  # Copy from project/frontend
```

2. **Fix packages/core imports**
```bash
# Update all components to use relative imports
# Example: '@/lib/types' → '../lib/types'
```

3. **Add missing configs to frontend/**
```bash
frontend/
├── tailwind.config.js  # Copy from project/frontend
├── postcss.config.js   # If needed
└── public/
    ├── vision_logo.png
    └── favicon.ico
```

### **Step 2: Test Everything** (CRITICAL)

```bash
# 1. Install dependencies
npm install

# 2. Type check all workspaces
npm run type-check

# 3. Build frontend
cd frontend && npm run build

# 4. Test Python package
cd python && pip install -e ".[dev,ai]"
python -m vision_forge.server

# 5. Integration test
# Start server + frontend dev mode
```

### **Step 3: Cleanup Redundant Files** (SAFE)

**Option A: Conservative Approach (Recommended)**
```bash
# Create a separate branch for cleanup
git checkout -b cleanup/remove-redundant-project-files

# Move project/ to project.legacy/
mv project project.legacy

# Update documentation to reference new structure only
# Test everything still works
# Create PR for review
```

**Option B: Aggressive Approach**
```bash
# Delete redundant files immediately
rm -rf project/frontend/dist
rm -rf project/frontend/node_modules
rm -rf project/frontend_build
rm -rf project/staticfiles
rm -rf project/backend/__pycache__
rm project/backend/db.sqlite3

# Keep only what's needed for web version:
# project/backend/          (Django for web)
# project/block_manager/    (Models for web)
```

### **Step 4: Create Web Version Transition Plan** (MEDIUM PRIORITY)

Create `WEB_VERSION_MIGRATION.md`:
```markdown
# Files to Move to visionforge-web Private Repo

## Backend (Django)
- project/backend/ → visionforge-web/backend/
- project/block_manager/ → visionforge-web/block_manager/

## Frontend (Web-Specific)
- project/frontend/src/lib/apiKeyContext.tsx
- project/frontend/src/lib/api.ts (adapt for Django)
- project/frontend/src/components/ApiKeyModal.tsx

## Create New in Web Repo
- src/lib/inference/api-client.ts (with API key support)
- package.json (depends on @visionforge/core from npm)
```

### **Step 5: Final Deployment Prep** (LOW PRIORITY)

1. **Add CI/CD workflows**
```yaml
.github/workflows/
├── test-core.yml       # Test packages/core
├── test-frontend.yml   # Test frontend build
├── test-python.yml     # Test Python package
└── publish-core.yml    # Publish to npm (manual trigger)
```

2. **Update README.md** with new structure
3. **Add badges** for build status, npm version, PyPI version
4. **Create CHANGELOG.md**

---

## 📊 Cleanup Impact Analysis

### Files to Keep in Public Repo
```
✅ packages/core/       - Shared UI components
✅ frontend/            - Local desktop frontend
✅ python/              - Python package
✅ docs/                - Documentation
✅ .github/             - CI/CD workflows
⚠️  project/backend/    - Keep until web repo created
⚠️  project/block_manager/ - Keep Django models for web
```

### Files to Remove from Public Repo (Eventually)
```
❌ project/frontend/          - Redundant with frontend/
❌ project/backend/           - Move to web repo
❌ project/block_manager/     - Move to web repo (keep services copy in python/)
❌ project/frontend_build/    - Old build artifacts
❌ project/staticfiles/       - Django static files
```

### Space Savings
```
Current:  18M project/
After:    ~8M (keep backend + block_manager for now)
          ~2M (after moving to web repo)

Savings:  ~16M (89% reduction)
```

---

## ⚡ Quick Win Checklist

Immediate tasks to make it deployment-ready:

- [ ] Copy `tailwind.config.js` to `frontend/`
- [ ] Copy `public/` assets to `frontend/public/`
- [ ] Create `frontend/src/lib/api.ts` for local client
- [ ] Fix import paths in `packages/core/src/components/`
- [ ] Add proper providers to `frontend/src/App.tsx`
- [ ] Run `npm install` and fix errors
- [ ] Test build: `cd frontend && npm run build`
- [ ] Test Python: `cd python && python -m vision_forge.server`
- [ ] Create `.github/workflows/test.yml`
- [ ] Update root README.md

---

## 🎯 Recommended Execution Order

### Week 1: Make It Work
1. Fix missing files (tailwind, api.ts, assets)
2. Fix import paths in packages/core
3. Install dependencies and fix errors
4. Test build pipeline
5. Document any issues

### Week 2: Clean It Up
1. Create cleanup branch
2. Remove redundant files (conservative approach)
3. Test everything still works
4. Update documentation
5. Create PR for review

### Week 3: Prepare for Deployment
1. Add CI/CD workflows
2. Test Python package installation
3. Create web version migration plan
4. Update README with deployment instructions
5. Tag v0.1.0-alpha

---

## 🚨 Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Breaking builds | HIGH | Test thoroughly before cleanup |
| Lost code | MEDIUM | Keep project.legacy/ backup |
| Import errors | HIGH | Fix incrementally, test each fix |
| Missing dependencies | MEDIUM | Document all required packages |
| Web version blocked | LOW | Keep project/ until web repo created |

---

Would you like me to start implementing any of these steps?
