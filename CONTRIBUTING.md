# Contributing to VisionForge

Thank you for your interest in contributing to VisionForge! This document provides guidelines and standards for contributing to the project.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branch & Commit Conventions](#branch--commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Code Standards](#code-standards)
- [Architecture Principles](#architecture-principles)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)

## Getting Started

VisionForge is a visual neural network architecture design tool with drag-and-drop block composition, automatic dimension inference, and PyTorch/TensorFlow code generation.

### Prerequisites
- **Frontend**: Node.js 18+, npm 9+
- **Backend**: Python 3.12+, pip
- Git with configured user name and email

### First-Time Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/visionforge.git
   cd visionforge
   ```

2. **Install frontend dependencies:**
   ```bash
   cd project/frontend
   npm install
   ```

3. **Install backend dependencies:**
   ```bash
   cd ../
   pip install -r requirements.txt
   ```

4. **Run the development servers:**
   ```bash
   # Frontend (from project/frontend/)
   npm run dev
   
   # Backend (from project/)
   python manage.py runserver
   ```

5. **Verify setup:**
   - Frontend should be running at http://localhost:5173
   - Backend should be running at http://localhost:8000

## Development Setup

### Project Structure
```
project/
  frontend/          # React SPA with TypeScript
    src/
      components/    # React components
      lib/          # Core logic (store, types, code gen, block defs)
      styles/       # Global CSS
  backend/          # Django REST API
    block_manager/  # Core app with services, views, models
```

### Running Tests
```bash
# Frontend
cd project/frontend
npm run lint
npm run build

# Backend
cd project
python manage.py test
```

## Branch & Commit Conventions

### Branch Naming
Use one of these prefixes:
- `feature/<short-kebab>` - New features or enhancements
- `bugfix/<ticket>` - Bug fixes
- `hotfix/<issue>` - Critical production fixes
- `chore/<scope>` - Maintenance, refactoring, or dependency updates

**Examples:**
- `feature/batch-normalization-layer`
- `bugfix/dimension-inference-concat`
- `chore/update-dependencies`

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `perf:` - Performance improvement
- `refactor:` - Code restructuring without behavior change
- `docs:` - Documentation updates
- `test:` - Test additions or updates
- `build:` - Build system or dependency changes
- `ci:` - CI/CD configuration changes
- `chore:` - Maintenance tasks

**Examples:**
```
feat: add Batch Normalization block with dimension inference

fix: resolve shape propagation error in concat merge blocks

docs: update NODES_AND_RULES.md with new loss functions
```

### Working with Git
1. **Keep your fork synced:**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Avoid merge commits** - use rebase to keep history linear
3. **Squash WIP commits** before opening PR - amend locally as needed
4. **Never force-push** to `main` or shared branches

## Pull Request Process

### Before Opening a PR - Self-Review Checklist

✅ **Code Quality:**
- [ ] Branch follows naming convention
- [ ] Rebased on latest `main` (no merge commits)
- [ ] Scope is focused (< ~300 lines if possible; split larger changes)
- [ ] Lint and type checks pass: `npm run lint`, `tsc --noEmit`
- [ ] No hardcoded secrets, API keys, or credentials

✅ **Testing:**
- [ ] Added/updated tests for new functionality
- [ ] Tests cover: shape inference, connection validation, code generation (if applicable)
- [ ] All tests pass locally

✅ **Documentation:**
- [ ] New block types documented in `docs/NODES_AND_RULES.md`
- [ ] Code generation changes include sample output
- [ ] Public APIs have JSDoc/docstring comments
- [ ] UI changes include screenshot or description

✅ **Architecture:**
- [ ] Follows established patterns (store actions, centralized validation)
- [ ] No duplicated logic (DRY principle)
- [ ] Proper separation of concerns (UI / store / domain logic / services)
- [ ] Changes are backward compatible (unless explicitly breaking)

### Opening Your PR

1. **Write a clear title** using conventional commit format
2. **Provide a detailed description:**
   - **Problem:** What issue does this solve?
   - **Solution:** How does your code address it?
   - **Testing:** What tests did you add/run?
   - **Risks:** Any potential breaking changes or edge cases?
   - **Screenshots:** For UI changes

3. **Link related issues** using keywords: `Fixes #123`, `Resolves #456`

### Review Process

1. **Automated Review:** Copilot agent will analyze your PR and post findings
2. **Human Review:** Maintainers will review for architecture, logic, and risk
3. **Feedback Categories:**
   - `BLOCKER:` - Must be fixed before merge
   - `SUGGEST:` - Optional improvement (can defer with rationale)
4. **Address feedback** - push new commits or amend existing ones
5. **Request re-review** once blockers are resolved
6. **Approval & Merge:**
   - Default: squash merge
   - Requires 1 approval (2 if touching core systems)
   - CI must be green

### Second Approval Required For:
- Changes to `inferDimensions` or connection validation logic
- Code generation pipeline modifications
- New external service integrations
- Security-related changes

## Code Standards

### TypeScript / Frontend

**General:**
- Strict TypeScript - avoid `any` unless justified
- Use Zustand store actions for all state mutations
- Components should be minimal/derived from store state
- Prefer functional components with hooks

**Styling:**
- Use CSS custom properties from `styles/theme.css`
- Tailwind for layout, custom properties for semantic colors
- Never hardcode colors - use `var(--color-primary)`, etc.

**Patterns:**
- Radix UI primitives (via `components/ui/`)
- Phosphor Icons for all icons
- Framer Motion for animations (spring physics)
- React Hook Form + Zod for form validation

**Example:**
```typescript
// ✅ Correct - use store action
const updateNode = useModelBuilderStore(state => state.updateNode);
updateNode(id, { config: { ...config, filters: 64 } });

// ❌ Wrong - direct mutation
node.data.config.filters = 64;
```

### Python / Backend

**General:**
- Python 3.12+ features encouraged
- Django REST Framework serializers for validation
- Type hints for all public functions
- Docstrings for non-trivial logic

**Security:**
- All external inputs validated via serializers
- No `eval` or `exec` of user code without sandboxing
- Secrets loaded from environment variables
- No credential logging

### Architecture Principles

Follow these core principles (detailed in `.github/copilot-instructions.md`):

1. **Modularity** - Single responsibility per file/component (~250 LOC max for mixed concerns)
2. **Loose Coupling** - Communicate via interfaces, not internal state
3. **High Cohesion** - Related logic stays together (e.g., shape logic in `blockDefinitions.ts`)
4. **Explicit Contracts** - Minimal public API surface, discriminated unions
5. **Readability > Cleverness** - Clear code over complex functional chains
6. **Extensibility** - New block types shouldn't require changes outside types/definitions/docs
7. **Testability** - Pure functions for transformations (no hidden dependencies)
8. **Deterministic** - Reproducible code generation and inference
9. **Layering** - UI → Store → Domain Logic → Services
10. **Error Surfacing** - Never swallow errors silently

### Anti-Patterns to Avoid

❌ Sprawling components mixing UI, logic, and data fetching
❌ Duplicated `computeOutputShape` logic outside block definitions
❌ Circular dependencies between modules
❌ Direct mutations outside store actions
❌ Tight coupling via deep internal imports
❌ Unbounded async operations without debounce/cancellation
❌ Environment-specific code in shared modules

## Testing Requirements

### When Tests Are Required

✅ **Always test:**
- New block types: `computeOutputShape` with normal + edge cases
- Connection validation changes
- Shape inference modifications
- Code generation alterations
- Backend validation/service logic

✅ **Test coverage should include:**
- Happy path (expected inputs)
- Edge cases (boundary values, empty inputs)
- Error conditions (invalid shapes, missing config)
- Regression cases (previously reported bugs)

### Frontend Testing
```bash
cd project/frontend
npm run test        # Run tests (when test framework is set up)
npm run lint        # ESLint
```

### Backend Testing
```bash
cd project
python manage.py test
python verify_nodes.py  # Verify block definitions
```

## Documentation

### When Documentation Updates Are Required

- **New block type** → Update `docs/NODES_AND_RULES.md`
- **Connection validation change** → Update node rules + README
- **New export format** → Update `frontend/EXPORT_FORMAT.md`
- **Architectural pattern** → Add ADR (Architectural Decision Record) in `docs/`
- **API changes** → Update `block_manager/documentation/API_REFERENCE.md`

### Documentation Standards

**Code Comments:**
- JSDoc for public TypeScript functions
- Docstrings for Python functions/classes
- Inline comments for complex algorithms only

**Markdown Files:**
- Clear headings and table of contents
- Code examples where applicable
- Keep up-to-date with code changes

### Architectural Decision Records (ADRs)

For significant architectural changes, create an ADR in `docs/`:

```markdown
# ADR-XXX: [Title]

## Context
What problem are we solving?

## Decision
What did we decide to do?

## Alternatives Considered
What other options did we evaluate?

## Consequences
What are the implications (positive and negative)?
```

## Community Guidelines

### Code of Conduct

- **Be respectful** - Treat all contributors with kindness and professionalism
- **Be constructive** - Focus feedback on code, not people
- **Be collaborative** - Help others learn and grow
- **Be patient** - Remember everyone was new once

### Review Etiquette

**For Authors:**
- Accept feedback graciously
- Ask questions if feedback is unclear
- Don't take criticism personally
- Update your PR promptly

**For Reviewers:**
- Be specific and actionable
- Use "Could we…" or "Consider…" for suggestions
- Prefix blockers clearly: `BLOCKER:` vs `SUGGEST:`
- Group related feedback to avoid comment spam
- Prioritize unblocking the author

### Getting Help

- **Questions?** Open a GitHub Discussion or issue
- **Bugs?** File an issue with reproduction steps
- **Ideas?** Start a Discussion to gather feedback first
- **Stuck?** Tag maintainers in your PR for guidance

### Recognition

We value all contributions:
- Code contributions (features, fixes, refactors)
- Documentation improvements
- Bug reports with clear reproduction steps
- Thoughtful PR reviews
- Helping others in discussions

Contributors are recognized in release notes and the project README.

## Merge Criteria Summary

Your PR will be merged when:
- ✅ All CI checks pass (lint, type check, tests)
- ✅ Required approvals received (1 or 2, depending on scope)
- ✅ All `BLOCKER:` comments resolved
- ✅ Documentation updated (if applicable)
- ✅ Tests added/updated (if applicable)
- ✅ No merge conflicts with `main`
- ✅ Follows code standards and architecture principles

### Blocking Conditions (PR will NOT be merged if):
- ❌ Undocumented public API or new block definition
- ❌ Silent failure paths or missing error handling
- ❌ Shape inference inconsistency
- ❌ Hard-coded secrets or credentials
- ❌ Introduces graph cycles without prevention
- ❌ Missing tests for new core logic

---

## Quick Reference

### Common Commands
```bash
# Start development
cd project/frontend && npm run dev

# Run linting
npm run lint
tsc --noEmit

# Rebase on main
git fetch origin
git rebase origin/main

# Run backend tests
cd project && python manage.py test
```

### Key Files to Know
- `project/frontend/src/lib/store.ts` - Zustand state management
- `project/frontend/src/lib/blockDefinitions.ts` - Block registry
- `project/frontend/src/lib/types.ts` - TypeScript type definitions
- `project/frontend/src/lib/codeGenerator.ts` - PyTorch/TF code generation
- `docs/NODES_AND_RULES.md` - Comprehensive node documentation
- `.github/copilot-instructions.md` - Detailed architecture & review standards

### Resources
- [Project README](./README.md)
- [PRD (Product Requirements Document)](./docs/PRD.md)
- [Nodes & Rules Documentation](./docs/NODES_AND_RULES.md)
- [Quick Start Guide](./docs/QUICKSTART.md)

---

**Thank you for contributing to VisionForge! 🚀**

Questions? Open an issue or start a discussion. We're here to help!
