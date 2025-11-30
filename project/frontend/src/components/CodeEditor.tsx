import { useMemo, useCallback } from 'react'
import CodeMirror, { ReactCodeMirrorProps } from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { EditorView } from '@codemirror/view'
import { keymap } from '@codemirror/view'
import { autocompletion } from '@codemirror/autocomplete'
import { useCodeMirrorTheme } from '@/hooks/use-codemirror-theme'
import { cn } from '@/lib/utils'

export interface CodeEditorProps {
  /** The code value to display/edit */
  value: string
  /** Callback when code changes (for editable mode) */
  onChange?: (value: string) => void
  /** Height of the editor (default: "80vh") */
  height?: string
  /** Whether the editor is read-only */
  readOnly?: boolean
  /** Whether the editor is editable (opposite of readOnly) */
  editable?: boolean
  /** Whether to show line numbers (default: true) */
  showLineNumbers?: boolean
  /** Whether to enable search panel (default: true) */
  enableSearch?: boolean
  /** Additional CSS classes for the container */
  className?: string
  /** Placeholder text when editor is empty */
  placeholder?: string
  /** Callback when user presses save shortcut (Ctrl/Cmd+S) */
  onSave?: () => void
  /** Callback when user presses escape */
  onEscape?: () => void
  /** ARIA label for accessibility */
  ariaLabel?: string
}

/**
 * Enhanced CodeMirror 6 editor component with theme integration and advanced features
 *
 * Features:
 * - Automatic theme switching (light/dark) synced with app theme
 * - Python syntax highlighting
 * - Auto-popup autocomplete
 * - Search/replace panel (Ctrl/Cmd+F)
 * - Line wrapping
 * - Code folding
 * - Keyboard shortcuts (Ctrl/Cmd+S to save, Esc to close)
 * - Accessibility support
 * - Muted background for read-only mode
 *
 * @example
 * ```tsx
 * // Read-only viewer
 * <CodeEditor
 *   value={code}
 *   readOnly={true}
 *   height="80vh"
 *   onEscape={handleClose}
 * />
 *
 * // Editable editor
 * <CodeEditor
 *   value={code}
 *   onChange={setCode}
 *   editable={true}
 *   height="60vh"
 *   onSave={handleSave}
 * />
 * ```
 */
export default function CodeEditor({
  value,
  onChange,
  height = '80vh',
  readOnly = false,
  editable = true,
  showLineNumbers = true,
  enableSearch = true,
  className,
  placeholder,
  onSave,
  onEscape,
  ariaLabel
}: CodeEditorProps) {
  // Theme integration with next-themes
  const theme = useCodeMirrorTheme()

  // Determine actual read-only state (readOnly takes precedence over editable)
  const isReadOnly = readOnly || !editable

  // Debounced onChange handler
  const debouncedOnChange = useCallback(
    (value: string) => {
      if (onChange && !isReadOnly) {
        // Debounce onChange to prevent excessive re-renders
        const timeoutId = setTimeout(() => {
          onChange(value)
        }, 0)
        // For immediate feedback, call onChange directly (no debounce for now)
        onChange(value)
        return () => clearTimeout(timeoutId)
      }
    },
    [onChange, isReadOnly]
  )

  // Custom keyboard shortcuts
  const customKeymap = useMemo(() => {
    const bindings: Array<{ key: string; run: () => boolean }> = []

    // Esc to close/escape
    if (onEscape) {
      bindings.push({
        key: 'Escape',
        run: () => {
          onEscape()
          return true
        }
      })
    }

    // Ctrl/Cmd+S to save (only in editable mode)
    if (onSave && !isReadOnly) {
      bindings.push({
        key: 'Mod-s',
        run: () => {
          onSave()
          return true // Prevent browser default save
        }
      })
    }

    return keymap.of(bindings)
  }, [onSave, onEscape, isReadOnly])

  // Configure all extensions
  const extensions = useMemo(() => {
    const exts = [
      python(), // Python syntax highlighting with built-in autocomplete
      EditorView.lineWrapping, // Line wrapping
      customKeymap // Custom keyboard shortcuts
    ]

    // Add autocomplete configuration for editable mode
    if (!isReadOnly) {
      exts.push(
        autocompletion({
          activateOnTyping: true, // Auto-popup as user types
          maxRenderedOptions: 10, // Show up to 10 suggestions
          defaultKeymap: true, // Enable Ctrl+Space trigger
          closeOnBlur: true // Close when editor loses focus
        })
      )
    }

    return exts
  }, [customKeymap, isReadOnly])

  // Basic setup configuration
  const basicSetup: ReactCodeMirrorProps['basicSetup'] = useMemo(
    () => ({
      lineNumbers: showLineNumbers,
      highlightActiveLineGutter: !isReadOnly, // Only in editable mode
      highlightActiveLine: !isReadOnly, // Only in editable mode
      foldGutter: true, // Code folding
      autocompletion: false, // We configure this manually via extensions
      closeBrackets: !isReadOnly, // Auto-close brackets only in editable mode
      bracketMatching: true, // Visual bracket matching
      searchKeymap: enableSearch, // Search panel
      completionKeymap: !isReadOnly, // Completion shortcuts
      lintKeymap: false // No linting
    }),
    [showLineNumbers, isReadOnly, enableSearch]
  )

  // Editor container classes with muted background for read-only
  const editorClasses = cn(
    'border rounded-md overflow-hidden',
    isReadOnly && 'bg-muted/30', // Subtle muted background for read-only
    className
  )

  // Render editor
  return (
    <div
      className={editorClasses}
      role="textbox"
      aria-label={ariaLabel || (isReadOnly ? 'Python code viewer' : 'Python code editor')}
      aria-readonly={isReadOnly}
    >
      <CodeMirror
        value={value}
        height={height}
        extensions={extensions}
        onChange={debouncedOnChange}
        editable={!isReadOnly}
        readOnly={isReadOnly}
        theme={theme}
        basicSetup={basicSetup}
        placeholder={placeholder}
      />
    </div>
  )
}
