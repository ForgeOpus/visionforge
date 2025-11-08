# Visual AI Model Builder - PRD

A browser-based visual interface for designing neural network architectures through intuitive drag-and-drop interactions, with automatic dimension inference, real-time validation, and PyTorch code generation.

**Experience Qualities:**
1. **Intuitive** - Building complex AI architectures should feel as natural as sketching on paper, with immediate visual feedback
2. **Empowering** - Users discover capabilities through exploration, with helpful guidance preventing mistakes before they happen
3. **Professional** - The interface conveys precision and technical sophistication appropriate for AI engineering work

**Complexity Level**: Light Application (multiple features with basic state)
This is a specialized design tool with persistent state management, real-time validation, and code generation - beyond a simple showcase but not requiring user accounts or backend infrastructure.

## Essential Features

### Canvas-Based Architecture Design
**Functionality**: Drag-and-drop neural network blocks onto an infinite canvas and connect them with visual edges
**Purpose**: Enable intuitive visual thinking about model architecture without code syntax barriers
**Trigger**: User drags a block from the palette onto the canvas
**Progression**: Select block from palette → Drag to canvas → Block appears with handles → Click handle → Drag to another block's handle → Connection validates → Dimensions auto-infer
**Success criteria**: Users can create complex multi-layer architectures in under 2 minutes; invalid connections prevented with clear explanations

### Real-Time Dimension Inference
**Functionality**: Automatically compute tensor shapes as blocks connect, propagating dimensions through the entire graph; supports multi-modal inputs defined as generalized tensor shapes
**Purpose**: Eliminate manual dimension calculation errors and provide instant feedback on architecture validity; enable flexibility for any data modality (text, image, audio, video, tabular)
**Trigger**: Connection created between two blocks or block parameter changed
**Progression**: Connection made → System traces back to input → Computes output shape based on layer parameters → Updates all downstream blocks → Visual display refreshes
**Success criteria**: All shape calculations complete within 100ms; users never see dimension mismatch errors at export time; supports arbitrary tensor dimensions for any modality

### Intelligent Block Configuration
**Functionality**: Context-aware parameter panels that show only relevant settings with smart defaults, inline validation, and quick presets for common modalities
**Purpose**: Guide users to correct configurations while allowing expert customization; provide quick-start templates for different data types
**Trigger**: User selects a block on canvas
**Progression**: Click block → Side panel opens → Display block-specific parameters → User modifies value or selects preset (Image/Text/Audio/Tabular) → Real-time validation → Dimensions update → Visual feedback
**Success criteria**: Required parameters clearly indicated; impossible values prevented; helpful tooltips on hover; one-click presets for common use cases

### Multi-Framework Code Export
**Functionality**: Generate complete, runnable PyTorch or TensorFlow model code with training boilerplate
**Purpose**: Bridge visual design to production-ready code without manual translation
**Trigger**: User clicks Export button after building valid architecture
**Progression**: Click export → Select framework → System validates architecture → Generates model class + training script + config → Display code preview → Copy to clipboard or download
**Success criteria**: Generated code runs without modification; includes helpful comments; follows framework best practices

### Project Persistence
**Functionality**: Save and load architecture designs with all configurations preserved
**Purpose**: Enable iterative design across sessions without losing work
**Trigger**: User clicks Save or loads a previous project
**Progression**: Click save → Architecture serialized to browser storage → Confirmation shown → Later: Click load → Select project → Canvas restores with all blocks, connections, and parameters
**Success criteria**: Projects persist indefinitely in browser; reload is pixel-perfect; supports 10+ saved projects

## Edge Case Handling

**Circular Dependencies** - Prevent connection creation that would create cycles; show tooltip "Neural networks must be acyclic"
**Orphaned Blocks** - Highlight unconnected blocks in orange; warning message "3 blocks not connected to main graph"
**Missing Input Block** - Prevent export with error "Architecture must start with an Input block"
**Dimension Mismatches** - Block connection attempt shows immediate tooltip "Conv2D requires 4D input [B,C,H,W], got 2D [B,F]"
**Browser Storage Limits** - Show warning at 80% quota; offer export to JSON file option
**Invalid Parameters** - Input field turns red; inline message "Must be positive integer"
**Multiple Frameworks** - Same visual architecture exports to different code syntax; internally track framework choice per project

## Design Direction

The design should feel like a precision engineering tool - clean, technical, and focused - with the polished sophistication of professional CAD software. A minimal interface ensures the architecture diagram remains the hero, while purposeful micro-interactions provide guidance without distraction. The aesthetic should communicate reliability and technical depth.

## Color Selection

**Triadic color scheme** (three equally spaced colors) creating visual hierarchy between input/processing/output operations, with each block category receiving a distinct hue family while maintaining harmony.

- **Primary Color**: Deep Blue (oklch(0.45 0.15 250)) - Represents core processing layers (Linear, Conv, etc.); conveys technical precision and computational intelligence
- **Secondary Colors**: 
  - Teal (oklch(0.55 0.12 180)) for input/output operations - suggests data flow and connectivity
  - Purple (oklch(0.50 0.13 290)) for advanced blocks (Attention, Transformer) - indicates sophisticated capabilities
- **Accent Color**: Vibrant Cyan (oklch(0.70 0.15 200)) for interactive elements, connection lines, and active states - creates visual energy and guides attention to actions
- **Foreground/Background Pairings**:
  - Background (Soft Gray oklch(0.98 0 0)): Dark text (oklch(0.20 0 0)) - Ratio 13.1:1 ✓
  - Card (White oklch(1 0 0)): Dark text (oklch(0.20 0 0)) - Ratio 14.8:1 ✓
  - Primary (Deep Blue oklch(0.45 0.15 250)): White text (oklch(1 0 0)) - Ratio 7.2:1 ✓
  - Secondary (Light Gray oklch(0.96 0 0)): Dark text (oklch(0.20 0 0)) - Ratio 12.8:1 ✓
  - Accent (Vibrant Cyan oklch(0.70 0.15 200)): White text (oklch(1 0 0)) - Ratio 4.9:1 ✓
  - Muted (Soft Gray oklch(0.95 0 0)): Muted text (oklch(0.50 0 0)) - Ratio 6.8:1 ✓

## Font Selection

Typography should balance technical legibility with modern sophistication - clear monospace numerals for dimensions, crisp sans-serif for labels, and consistent hierarchy throughout the interface. **Inter** for all UI text (exceptional clarity at small sizes, professional feel) and **JetBrains Mono** for code display and dimension annotations (designed for programming contexts).

- **Typographic Hierarchy**: 
  - H1 (Project Title): Inter SemiBold/24px/tight tracking - strong presence without overwhelming
  - H2 (Panel Headers): Inter Medium/16px/normal tracking - clear section delineation  
  - Body (Block Labels): Inter Regular/14px/normal tracking - optimal legibility on canvas
  - Small (Dimensions): JetBrains Mono Regular/12px/wide tracking - technical precision
  - Code (Export Preview): JetBrains Mono Regular/13px/normal tracking - familiar to developers

## Animations

Animations should reinforce the sense of a living, responsive system - blocks settle into place with subtle physics, connections draw with purpose, and validation feedback appears instantly but gracefully. Movement is restrained and functional, never decorative.

- **Purposeful Meaning**: Blocks "snap" into grid alignment with gentle spring physics, communicating the underlying structure; connection lines draw from source to target (not fade in) to show direction of data flow; validation errors pulse once to catch attention without distraction
- **Hierarchy of Movement**: Connection creation (300ms ease-out) receives most animation emphasis as it's the primary creative action; block selection is instant (50ms) for responsive feel; panel transitions are quick (200ms) to feel snappy; validation feedback appears immediately (100ms) then settles

## Component Selection

- **Components**: 
  - Canvas: Custom div-based infinite canvas with pan/zoom controls, SVG overlay for connections
  - Block Palette: ScrollArea with categorized Accordion sections, each BlockItem as draggable Card
  - Config Panel: Sheet (right-anchored) with dynamic Form components based on selected block
  - Blocks: Custom Card components with gradient borders indicating category, Badge for block type
  - Connections: Custom SVG paths with Arrow markers, color-coded by validation state
  - Export Modal: Dialog with Tabs for PyTorch/TensorFlow, syntax-highlighted code in ScrollArea
  - Inputs: Input for text/numbers, Select for dropdowns, Switch for booleans, Slider for ranges
  - Validation: Alert components for errors, Toast (sonner) for save/load confirmations
  - Project Selector: DropdownMenu in header with saved projects list
  
- **Customizations**: 
  - Block cards need custom drag handles and connection ports (small circles on edges)
  - Connection SVG paths need animated drawing effect on creation
  - Config panel forms need dynamic rendering based on block type schema
  - Canvas needs custom zoom controls (+ - reset buttons) in bottom-right
  - Block palette items show icon + name in horizontal layout with drag cursor
  
- **States**: 
  - Blocks: default (neutral), selected (cyan border + shadow), invalid (red border), dragging (50% opacity)
  - Connections: valid (cyan solid line), invalid-prevented (red dashed), hover (thicker line)
  - Inputs: default, focused (cyan ring), error (red border + text), disabled (gray)
  - Buttons: default, hover (slight scale + brightness), active (pressed scale), loading (spinner)
  
- **Icon Selection**: 
  - Phosphor icons throughout for consistency and clarity
  - FlowArrow for connections, Plus for add block, Download for export
  - Lightning for processing blocks, Brain for AI blocks, GitBranch for splits/merges
  - Eye for visibility toggle, Trash for delete, Copy for duplicate
  - Warning for validation errors, CheckCircle for success states
  
- **Spacing**: 
  - Canvas grid: 20px for subtle alignment guidance
  - Panel padding: p-6 for breathing room around content
  - Block internal padding: p-4 for compact but readable
  - Form field spacing: space-y-4 for clear field separation
  - Button spacing: px-6 py-2 for comfortable touch targets
  - Section gaps: gap-8 between major sections, gap-4 within sections
  
- **Mobile**: 
  - Stack layout: Canvas goes full-screen with floating action buttons
  - Block palette: Bottom sheet drawer (swipe up to open)
  - Config panel: Full-screen modal overlay when block selected
  - Touch targets: Minimum 44px for all interactive elements
  - Gestures: Pinch-to-zoom on canvas, long-press to select, double-tap for properties
  - Simplified: Hide dimension annotations until block selected (reduce visual clutter)
  - Progressive: Desktop shows all three panels simultaneously; mobile shows one at a time
