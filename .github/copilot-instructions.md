# VisionForge AI Model Builder - Copilot Instructions

## Project Overview
VisionForge is a visual neural network architecture design tool with drag-and-drop block composition, automatic dimension inference, and PyTorch code generation. It's a monorepo with a React/TypeScript frontend and Django backend (currently minimal).

**Tech Stack:**
- **Frontend**: React 19, TypeScript, Vite, @xyflow/react (canvas), Zustand (state), Radix UI + Tailwind CSS
- **Backend**: Django 5.2, Python 3.12+ (currently scaffolded but minimal)
- **Key Dependencies**: @github/spark, framer-motion, react-hook-form, zod

## Architecture & Data Flow

### State Management (Zustand)
- **Single store**: `src/lib/store.ts` - all application state lives here
- State structure: `nodes`, `edges`, `selectedNodeId`, `validationErrors`, `currentProject`
- **Key pattern**: State mutations trigger `inferDimensions()` to propagate tensor shapes through the graph
- No Redux, no Context API - Zustand hooks like `useModelBuilderStore()` are the only state access pattern

### Block System
1. **Definitions**: `src/lib/blockDefinitions.ts` - registry of all available neural network blocks
   - Each block has: `type`, `label`, `category`, `color`, `icon`, `configSchema`, `computeOutputShape()`
   - Categories: `input`, `basic`, `advanced`, `merge`
   - **Connection validation**: `validateBlockConnection()` enforces dimension compatibility rules
   - **Multi-input support**: `allowsMultipleInputs()` identifies merge blocks (concat, add)
2. **Type System**: `src/lib/types.ts` - strict TypeScript interfaces for `BlockData`, `TensorShape`, `BlockConfig`
3. **Dimension Inference**: Automatic shape propagation via `inferDimensions()` in store
   - Walks graph from input nodes, computes output shapes using `computeOutputShape()` functions
   - Updates triggered on node addition, edge creation, or config changes
4. **Custom Layers**: Special handling with modal dialog (`CustomLayerModal.tsx`) for code editing
   - Uses CodeMirror (@uiw/react-codemirror) for Python syntax highlighting
   - Configuration shown in modal, not sidebar
   - User can write custom forward pass logic

### Canvas & Flow
- Uses **@xyflow/react** for node-based editor (not React Flow - this is XYFlow)
- Custom node component: `BlockNode.tsx` renders blocks with colored borders, shape annotations
- Connection validation: `validateConnection()` prevents invalid connections (e.g., Conv2D requires 4D input)
- **Pattern**: Canvas operations go through store actions, never direct ReactFlow state manipulation

### Code Generation
- `src/lib/codeGenerator.ts` generates complete PyTorch projects (model.py, train.py, config.py)
- Topological sort for layer ordering, then translate block configs to PyTorch syntax
- Currently PyTorch-only (TensorFlow generation stubbed)

## Development Workflows

### Running the App
```bash
# Frontend (from project/frontend/)
npm run dev          # Vite dev server on port 5173
npm run build        # Production build
npm run lint         # ESLint

# Backend (from project/)
python manage.py runserver  # Django dev server on port 8000 (not currently used)
```

### Adding a New Block Type
1. Add type to `BlockType` union in `types.ts`
2. Create definition in `blockDefinitions.ts` with `computeOutputShape()` logic
3. No UI changes needed - palette and config panel auto-generate from schema

### State Mutations
**Always** use store actions - never mutate nodes/edges directly:
```typescript
// ✅ Correct
updateNode(id, { config: { ...config, param: value } })

// ❌ Wrong
node.data.config.param = value
```

## Project-Specific Conventions

### Styling
- **Triadic color scheme**: Deep Blue (primary), Teal (input/output), Purple (advanced), Cyan (accent)
- Colors defined as CSS custom properties in `styles/theme.css`
- Use `var(--color-primary)`, `var(--color-accent)`, etc. - never hardcode colors
- Tailwind for layout, custom properties for semantic colors

### Component Patterns
- Radix UI primitives wrapped in `components/ui/` (shadcn/ui style)
- Always use `forwardRef` for UI components that accept refs
- Phosphor Icons (`@phosphor-icons/react`) for all icons
- Framer Motion for animations (spring physics, not easing curves)

### Type Safety
- Strict TypeScript - no `any` without explicit justification
- Zod schemas for form validation (see `react-hook-form` + `@hookform/resolvers`)
- Block configs are type-safe via `BlockConfig` interface

### Error Handling
- Validation errors stored in `validationErrors` array with `nodeId`, `message`, `type` fields
- Toast notifications via `sonner` library
- No try-catch without logging - errors should be visible to users

## Integration Points

### API (Currently Unused)
- `src/lib/api.ts` defines API client for backend
- Backend endpoints planned: `/api/validate`, `/api/chat`, `/api/export`
- **Current state**: All operations are client-side; backend is scaffolded but not integrated

### File Structure
```
project/
  frontend/           # React SPA
    src/
      components/     # React components
        ui/          # Radix UI wrappers
      lib/           # Core logic (store, types, code gen, block defs)
      styles/        # Global CSS
  backend/           # Django (minimal, not actively used)
```

### Storage
- Projects are in-memory only (no localStorage/backend persistence yet)
- `Project` type includes `nodes`, `edges`, `createdAt`, `updatedAt` but saving is not persisted

## Critical Patterns

1. **Dimension inference is automatic** - adding edges or changing configs triggers recalculation
2. **Block definitions drive UI** - config panel and palette auto-generate from `configSchema`
3. **Zustand is the source of truth** - component state should be minimal/derived
4. **XYFlow, not React Flow** - use `@xyflow/react` imports, not `reactflow`
5. **Theme colors via CSS properties** - respect design system defined in PRD (docs/PRD.md)
6. **Connection rules are enforced** - see `validateBlockConnection()` for dimension compatibility
7. **Node documentation** - comprehensive rules in `docs/NODES_AND_RULES.md`

## Known Gotchas

- Multi-input blocks (`concat`, `add`) require special handling in `validateConnection()`
- Connection validation enforces strict dimension rules (e.g., Conv2D needs 4D, Linear needs 2D)
- Circular dependency detection is critical - prevent cycles in graph
- Shape validation happens at connection time, not render time
- Linear layers require 2D input - auto-suggest Flatten if input is 4D
- Custom blocks use modal dialog for configuration, not sidebar - check `blockType === 'custom'` in ConfigPanel
- Custom layer code is stored in block config and displayed with syntax highlighting


## Code Review & PR Governance
**Goal:** Maintain high velocity without sacrificing correctness, security, or maintainability. Reviews focus on material risk or user-impact changes—never cosmetic nitpicks already enforced by automated tooling.

### Roles
**Author:** Produces clear, minimal, well‑scoped PRs and updates related docs/tests.
**Reviewer:** Evaluates correctness, risk, and alignment with architecture patterns; provides concise, actionable feedback.
**Copilot Agent:** Automates diff parsing, highlights only high‑signal issues (security, breaking change, logic error, missing tests/docs), and suggests concrete remediation steps.

### PR Author Checklist (must self‑verify before requesting review)
1. Branch name follows pattern: `feature/<short-kebab>`, `bugfix/<ticket>`, `hotfix/<issue>`, `chore/<scope>`.
2. Rebased on latest `main` (no merge commits; use `git fetch; git rebase origin/main`).
3. Scope is single concern (if > ~300 lines net change across disparate areas, split).
4. All new block types update `docs/NODES_AND_RULES.md` and any quick references.
5. Code generation changes include sample output validation (attach snippet or test).
6. Lint + type check pass locally: `npm run lint` / `tsc --noEmit` and relevant Python checks if touched.
7. Added/updated tests for: shape inference edge cases, connection validation, code generation ordering, or new services.
8. Secrets/config remain external (no tokens, API keys, or `.env` contents committed).
9. UI changes include screenshot or short description of impact (accessibility / layout).
10. Changelog stub added if change affects generated artifacts or user workflow.

### Commit & Branch Conventions
- Prefer Conventional Commits: `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `build:`, `ci:`, `chore:`.
- Squash on merge (default) to keep history linear; preserve multiple commits only for complex debugging narratives.
- Avoid WIP commits in PR history—use local/amended commits.

### Labeling & Prioritization
Use labels to accelerate triage:
- `type:feature`, `type:bug`, `type:security`, `type:refactor`, `type:docs`.
- `area:frontend`, `area:backend`, `area:codegen`, `area:infra`, `area:validation`.
- Severity: `sev:critical` (security/data loss), `sev:high` (core logic regression), `sev:normal`, `sev:low`.
- Copilot agent may auto‑suggest labels; human reviewer confirms.

### Review Workflow
1. Author opens PR with clear description: problem, solution approach, risks, test evidence.
2. Copilot agent runs automated review (diff + heuristics) and posts summary comment.
3. Human reviewer(s) skim agent output, then review code—focus on architectural alignment and nuanced logic.
4. Reviewer marks blockers vs. optional suggestions (prefix comments with `BLOCKER:` or `SUGGEST:`).
5. Author addresses blockers; optional suggestions may be deferred with rationale.
6. Final pass: ensure no stale conversations. Reviewer approves; second approval required if:
   - Touches core inference (`inferDimensions`, connection validation)
   - Modifies code generation pipeline
   - Introduces new external service integration
7. Merge (squash) once CI green, required approvals present, and no open blockers.

### Blocking Conditions (Do NOT merge if any are present)
- Undocumented public API or new block definition missing docs.
- Silent failure paths (empty catch / missing error handling on external boundary).
- Shape inference inconsistency (output shape mismatch vs. documented contract).
- Hard‑coded secrets or credentials.
- Introduces cycle in graph processing without explicit prevention logic.
- Missing tests for new computeOutputShape logic or dimension edge cases.

### Security & Compliance Review Checklist
For changes touching backend, code generation, or external AI providers:
- Input validation: All external inputs either schema‑validated (Zod / DRF serializer).
- No dynamic `eval`/`exec` of user code without sandboxing (custom layer code must remain server‑side isolated if later executed).
- Dependency additions reviewed (no abandoned / low trust packages).
- Secrets: loaded via environment; no logging of credential values.
- Model/code export: ensure no injection vectors in generated Python (sanitize custom layer identifiers).
- Serialization: confirm no unsafe pickle usage; prefer JSON or deterministic formats.

### Performance Considerations
- Flag PRs that increase average graph traversal complexity (e.g., unnecessary recomputation loops).
- Large canvas interaction changes must avoid excessive re-renders (watch for expanded Zustand selectors).
- Code generation: ensure topological sort remains O(n + e); no quadratic passes added.

### Test Coverage Guidance
- New block: tests for `computeOutputShape` (normal + edge input) and connection validation failure modes.
- Inference changes: regression test ensuring shapes propagate after config mutation & edge addition.
- Code generation alterations: snapshot or assertion of layer ordering + emitted class structure.
- Backend service logic: tests mock external providers—cover success + error path.

### Documentation Triggers
- Add/modify block type ⇒ update `docs/NODES_AND_RULES.md` + quick references.
- Change to connection validation rules ⇒ highlight in README + node rules doc.
- New export format ⇒ update `frontend/EXPORT_FORMAT.md`.

### Merge Policy
- No merge commits from `main` after review begins—rebase instead.
- PRs idle > 7 days: re‑validate against `main`; Copilot agent refreshes review summary.

### Copilot Agent Review Protocol
1. Parse diff and classify changes (frontend logic, backend logic, codegen, docs).
2. Generate concise summary: impacted modules + potential risk hotspots.
3. Emit prioritized checklist (max 8 items) with severity tags.
4. Avoid stylistic feedback; ignore whitespace‑only diffs.
5. Suggest minimal test additions when coverage gaps detected.
6. If no material issues: post “LGTM (no blockers)” summary.

### Changelog & Versioning
- Functional change to generated model/code ⇒ minor version bump.
- Block definition addition/removal ⇒ minor bump.
- Pure bug fix or docs ⇒ patch bump.
- Security fix ⇒ patch bump + `sev:critical` label.

### When to Request Additional Tests
Request tests if PR introduces:
- New branching logic in dimension inference.
- Multi‑input merge behavior changes.
- Non‑trivial transformation in code generation.
- Backend data validation paths not previously covered.

### Non-Goals for Review
- Styling already enforced by Tailwind + linting.
- Micro-optimizations without benchmarks.
- Subjective naming unless ambiguous or misleading.

### Reviewer Tone & Conduct
- Be specific, respectful, and prioritize unblock time.
- Prefer “Could we…” or “Consider…” for non-blocking suggestions.
- Group related feedback; avoid comment spam.

### Summary
Focus on correctness, stability, and user impact. Automate the rote; apply human judgment to architecture, clarity, and risk.

***

**Do not:**
- Nitpick variable names, whitespace, or stylistic choices already enforced by automated tools.
- Repeat feedback already caught by linters, formatters, or standard unit tests.

## Architecture & Maintainability Standards
**Objective:** Every contribution should strengthen long‑term evolvability: modular, readable, testable, performant, and loosely coupled. Copilot must proactively surface architectural risks and recommend concrete refactors.

### Core Principles
1. Modularity: Prefer cohesive files/components with single responsibility. Split when a component exceeds ~250 LOC of mixed concerns (UI + data + side‑effects).
2. Loose Coupling: Cross‑module communication via clear interfaces (TypeScript types, service functions, Django serializers). Avoid reaching into another module's internals or Zustand state slices directly—use store actions.
3. High Cohesion: Related logic resides together (e.g., block shape calculation logic stays in `blockDefinitions.ts`, not scattered across UI components).
4. Explicit Contracts: Public functions/types should expose minimal necessary surface; use discriminated unions and exact object types for clarity.
5. Readability > Cleverness: Prefer clear loops over nested functional chains when shape or state mutation reasoning matters.
6. Extensibility: Adding a new block type should never require changes outside `types.ts`, `blockDefinitions.ts`, and docs; flag if new code introduces extra touchpoints.
7. Testability: Logic performing transformation (shape inference, code generation sorting, validation) should remain pure and callable in isolation (no direct DOM / network / random dependencies).
8. Deterministic Behavior: Code generation & inference must be reproducible—avoid hidden state, timestamps in generated code, or side effects during pure computations.
9. Layering: UI (components) → orchestration (store/actions) → domain logic (lib/* utilities) → integration (backend services). Copilot must flag violations (e.g., UI calling backend service directly bypassing store or domain layer).
10. Error Surfacing: Never swallow errors; centralize user‑visible validation in store error arrays or explicit toast pathways.

### Required Patterns to Preserve
- Store is source of truth; no duplicated shape logic in components.
- Connection validation rules centralized (no shadow logic in UI preventing edges silently).
- Code generation passes: collect blocks → topological sort → emit layer code → build training harness; maintain this sequence.
- Custom layer safety: treat user code as opaque text—do not parse/execute client side beyond syntax highlight.

### Anti‑Patterns to Flag (Copilot Agent)
- Sprawling component with shape inference, rendering, and networking combined.
- Any duplication of `computeOutputShape` logic outside registered definitions.
- Implicit circular dependencies (e.g., lib util importing component which imports store which imports that util again).
- Hidden mutations of node config outside `updateNode()`.
- Tight coupling: direct imports from deep internal paths of unrelated feature folders.
- Unbounded async chains without cancellation (e.g., validating on every keystroke without debounce).
- Adding environment‑specific code (e.g., Node APIs) inside shared browser modules.

### Copilot Agent Detailed Feedback Protocol
When architectural issues are detected in a PR:
1. Classify issue (e.g., Coupling, Cohesion, SRP Violation, Hidden Side Effect, Contract Ambiguity, Testability Gap).
2. Provide a 2–4 sentence explanation of impact (maintenance risk, regression likelihood, performance side effect, onboarding friction).
3. Suggest a concrete remediation path: files to split, functions to extract, interface/type to introduce, test to add.
4. If refactor touches critical logic (`inferDimensions`, connection validation, code generation sort) recommend adding/adjusting tests and list specific edge cases.
5. Link (inline citation) to principle above (e.g., "Violates Modularity Principle #1").
6. Prioritize issues: High (must fix pre‑merge), Medium (fix or justify), Low (optional improvement). Use these tags.
7. Avoid abstract advice—always reference exact symbols / file paths.

### Maintainability Review Checklist (Author Self‑Check)
- Does each file have a single dominant reason to change?
- Can new block types be added without modifying existing component logic?
- Are all shape/inference changes centrally located?
- Are service integrations isolated under `services/` with clear factory patterns?
- Are public types exported intentionally (no leaking internal helper types)?
- Are complex conditions replaced by descriptive helper functions?
- Is dead code / commented‑out legacy removed?

### Refactor Triggers
Trigger a refactor (or raise a review comment) when:
- Function exceeds ~60 LOC with mixed concerns.
- Duplicate logic appears 3+ times (prefer utility extraction).
- Introducing a second conditional path requiring knowledge of internal store state shape.
- Adding a new dependency solely to perform trivial formatting; prefer existing stack.

### Extension Safety
Ensure extensibility for future backends (TensorFlow full support, ONNX export) by:
- Keeping backend‑agnostic representations of model graphs pure.
- Isolating framework‑specific code under `services/pytorch_codegen.py` and `services/tensorflow_codegen.py`.
- Avoiding hard‑coded framework strings in UI layer—use enums/config maps.

### Documentation Expectations
Non-trivial public utilities must include a brief JSDoc or docstring explaining purpose & constraints.
Any new architectural pattern (e.g., plugin loader) requires a `docs/` entry summarizing rationale & usage.

### Architectural Decision Records (ADRs)
For significant changes (new code generation pass, validation engine redesign) add a markdown file under `docs/` with: Context, Decision, Alternatives, Consequences.

### Copilot Tone & Depth
Copilot should balance brevity with precision: short summary, then rich detail only for High priority issues. Provide before/after pseudo-code where refactor suggested.

---
**In summary:** Architecture feedback is not optional; it must be actionable, prioritized, and mapped to explicit principles to accelerate correct fixes.
