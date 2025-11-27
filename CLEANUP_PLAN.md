# VisionForge - Cleanup & Optimization Plan

## 🎯 Goal
Transform the current hybrid structure into a clean, deployment-ready monorepo by removing redundant code while maintaining backward compatibility for the web version.

---

## 📊 Current State

### Directory Structure
```
visionforge/
├── 18M   project/              # 97% of repo size - MOSTLY REDUNDANT
│   ├── backend/                # Django - move to web repo
│   ├── block_manager/          # Services copied to python/
│   └── frontend/               # Frontend copied to packages/core + frontend/
│
├── 460K  packages/core/        # ✅ NEW - Shared components
├── 50K   frontend/             # ✅ NEW - Local desktop frontend
└── 495K  python/               # ✅ NEW - Python package
```

### Redundancy Analysis

#### 🔴 High Redundancy (90%+ duplicate)
- `project/frontend/src/lib/` → Copied to `packages/core/src/lib/`
- `project/frontend/src/components/` → Copied to `packages/core/src/components/`
- `project/block_manager/services/` → Copied to `python/vision_forge/services/`

#### 🟡 Partial Redundancy (Some web-specific code)
- `project/frontend/src/` → Has `apiKeyContext.tsx`, `ApiKeyModal.tsx` (web-only)
- `project/backend/` → Needed for web version (should move to private repo)

#### 🟢 No Redundancy (Keep as-is)
- `packages/core/` - New shared package
- `frontend/` - New local frontend
- `python/` - New Python package
- `docs/` - Documentation

---

## 🗂️ File-by-File Cleanup Strategy

### Phase 1: Safe Deletions (No Impact)

**Build Artifacts & Dependencies**
```bash
# Frontend build outputs
rm -rf project/frontend/dist/
rm -rf project/frontend/node_modules/
rm -rf project/frontend/.vite/

# Django build outputs
rm -rf project/frontend_build/
rm -rf project/staticfiles/

# Python cache
find project/ -type d -name "__pycache__" -exec rm -rf {} +
find project/ -name "*.pyc" -delete
find project/ -name "*.pyo" -delete

# Database (keep .sqlite3 for now, but can delete for production)
rm -f project/backend/db.sqlite3
rm -f project/db.sqlite3

# IDE files
rm -rf project/.vscode/
rm -rf project/.idea/
```

**Space saved: ~5-8MB**

### Phase 2: Mark as Deprecated (Add README)

Create `project/README_DEPRECATED.md`:
```markdown
# ⚠️ DEPRECATED STRUCTURE

This directory contains the legacy monolithic structure.

## Status
- **Frontend**: Migrated to `packages/core/` + `frontend/`
- **Backend**: To be moved to private `visionforge-web` repository
- **Services**: Copied to `python/vision_forge/services/`

## What to Use Instead

### For Local Development
- Use `frontend/` for local desktop app
- Use `python/` for Python package
- Use `packages/core/` for shared components

### For Web Development
This structure will be moved to a private repository.

## Timeline
- [ ] Create private visionforge-web repository
- [ ] Move Django backend to web repo
- [ ] Move web-specific frontend code to web repo
- [ ] Delete this directory

## DO NOT ADD NEW CODE HERE
All new development should happen in:
- `packages/core/` - Shared components
- `frontend/` - Local desktop app
- `python/` - Python package
```

### Phase 3: Restructure for Web Migration

**Create migration manifest:**
```bash
project/WEB_MIGRATION_MANIFEST.txt
```

```
# Files to Move to visionforge-web Private Repository

## Backend (Complete Django Project)
project/backend/ → visionforge-web/backend/
project/block_manager/ → visionforge-web/block_manager/
project/manage.py → visionforge-web/manage.py
project/requirements.txt → visionforge-web/requirements.txt
project/pyproject.toml → visionforge-web/pyproject.toml

## Frontend Web-Specific Files
project/frontend/src/lib/apiKeyContext.tsx → visionforge-web/src/lib/
project/frontend/src/lib/api.ts → visionforge-web/src/lib/api.ts (adapt)
project/frontend/src/components/ApiKeyModal.tsx → visionforge-web/src/components/

## Environment & Config
project/.env.example → visionforge-web/.env.example
project/frontend/vite.config.ts → visionforge-web/vite.config.ts (adapt)

## Build Scripts
project/build_frontend.sh → visionforge-web/build_frontend.sh
project/build_frontend.bat → visionforge-web/build_frontend.bat
```

### Phase 4: Archive Redundant Code

Instead of deleting, create archive:

```bash
# Create archive branch
git checkout -b archive/project-legacy
git mv project project.legacy
git commit -m "Archive legacy project structure"
git push origin archive/project-legacy

# Back to main branch - project/ removed
git checkout main
```

Or create tarball:
```bash
tar -czf project-legacy-$(date +%Y%m%d).tar.gz project/
# Upload to releases or S3
rm -rf project/
```

---

## 📋 Step-by-Step Cleanup Script

### Option A: Conservative (Recommended)

```bash
#!/bin/bash
# cleanup-conservative.sh

echo "🧹 VisionForge Conservative Cleanup"
echo "===================================="

# Step 1: Delete build artifacts only
echo "Step 1: Removing build artifacts..."
rm -rf project/frontend/dist
rm -rf project/frontend/node_modules
rm -rf project/frontend/.vite
rm -rf project/frontend_build
rm -rf project/staticfiles
find project/ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find project/ -name "*.pyc" -delete
find project/ -name "*.pyo" -delete

# Step 2: Mark as deprecated
echo "Step 2: Marking project/ as deprecated..."
cat > project/README_DEPRECATED.md <<'EOF'
# ⚠️ DEPRECATED STRUCTURE

This directory is deprecated. Use:
- `packages/core/` for shared components
- `frontend/` for local desktop app
- `python/` for Python package

This will be moved to private visionforge-web repository.
EOF

# Step 3: Show results
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Space saved:"
du -sh project/ 2>/dev/null
echo ""
echo "Next steps:"
echo "1. Test everything still works"
echo "2. Create private web repository"
echo "3. Run cleanup-aggressive.sh to remove project/"
```

### Option B: Aggressive (After Web Repo Created)

```bash
#!/bin/bash
# cleanup-aggressive.sh

echo "🧹 VisionForge Aggressive Cleanup"
echo "=================================="
echo "⚠️  WARNING: This will remove project/ directory"
echo ""
read -p "Have you created visionforge-web repository? (y/N): " confirm

if [ "$confirm" != "y" ]; then
    echo "❌ Aborted. Create web repository first."
    exit 1
fi

# Create archive first
echo "Creating archive..."
tar -czf project-archive-$(date +%Y%m%d).tar.gz project/
echo "✅ Archive created: project-archive-$(date +%Y%m%d).tar.gz"

# Remove project directory
echo "Removing project/..."
rm -rf project/

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Changes:"
echo "- Removed: project/"
echo "- Created: project-archive-*.tar.gz"
echo ""
echo "Next steps:"
echo "1. git add -A"
echo "2. git commit -m 'Remove legacy project structure'"
echo "3. git push"
```

---

## 🔍 What NOT to Delete

### Keep Until Web Repo Created
```
project/backend/              # Django backend for web version
project/block_manager/        # Django models (different from services)
project/manage.py             # Django entry point
project/.env.example          # Environment template
```

### Keep Forever (Reference)
```
docs/                         # Documentation
project/verify_nodes.py       # Useful utility
```

### Already Gitignored (Auto-cleaned)
```
node_modules/
__pycache__/
*.pyc
dist/
.vite/
db.sqlite3
```

---

## 📈 Expected Results

### Before Cleanup
```
Total repository size: ~20MB
- project/: 18MB (90%)
- New structure: 2MB (10%)
```

### After Conservative Cleanup
```
Total repository size: ~12MB
- project/: 10MB (kept backend + block_manager)
- New structure: 2MB
Space saved: 8MB (40%)
```

### After Aggressive Cleanup (Web repo created)
```
Total repository size: ~2MB
- New structure only: 2MB (100%)
Space saved: 18MB (90%)
```

---

## ⚠️ Risks & Safeguards

### Risk 1: Breaking Web Version
**Mitigation:**
- Keep `project/` until web repository is created
- Test web version before deleting
- Create archive/tarball before deletion

### Risk 2: Lost Code
**Mitigation:**
- Use git branches for cleanup
- Create archives before deletion
- Document what was removed and why

### Risk 3: Import Errors
**Mitigation:**
- Update imports incrementally
- Test after each change
- Keep rollback branch

### Risk 4: Missing Dependencies
**Mitigation:**
- Document all dependencies in new package.json
- Test installation from scratch
- Keep old package.json for reference

---

## 🎬 Recommended Execution Plan

### Timeline: 2 Weeks

**Week 1: Prepare**
- [ ] Run conservative cleanup script
- [ ] Test local version still works
- [ ] Document all project/ dependencies
- [ ] Create web migration manifest
- [ ] Get approval for aggressive cleanup

**Week 2: Execute**
- [ ] Create private visionforge-web repository
- [ ] Migrate Django backend to web repo
- [ ] Migrate web-specific frontend to web repo
- [ ] Test web version independently
- [ ] Run aggressive cleanup script
- [ ] Test local version without project/
- [ ] Commit and push cleanup

---

## 🚀 Quick Start

To begin cleanup now:

```bash
# 1. Create cleanup script
cat > cleanup.sh << 'EOF'
#!/bin/bash
rm -rf project/frontend/dist project/frontend/node_modules
rm -rf project/frontend_build project/staticfiles
find project/ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find project/ -name "*.pyc" -delete
echo "✅ Build artifacts removed"
EOF

# 2. Make executable
chmod +x cleanup.sh

# 3. Run cleanup
./cleanup.sh

# 4. Check results
du -sh project/

# 5. Commit
git add -A
git commit -m "Remove build artifacts from legacy project structure"
git push
```

---

## 📝 Summary

**Safe to Delete Now:**
- Build artifacts (dist/, node_modules/, __pycache__/)
- Database files (db.sqlite3)
- Cache files (.vite/, .pyc)

**Delete After Web Repo Created:**
- project/frontend/ (redundant with packages/core + frontend/)
- project/backend/ (move to web repo)
- project/block_manager/ (move to web repo)

**Never Delete:**
- packages/core/
- frontend/
- python/
- Documentation

**Result:** Clean, maintainable monorepo focused on local desktop version, with web version in separate private repository.
