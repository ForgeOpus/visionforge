import { useMemo } from 'react'
import { useTheme } from 'next-themes'
import { githubLight, githubDark } from '@uiw/codemirror-theme-github'
import type { Extension } from '@codemirror/state'

/**
 * Custom hook for CodeMirror theme integration with next-themes
 *
 * Returns the appropriate GitHub theme (light or dark) based on the current app theme.
 * Automatically syncs with theme changes and handles SSR/hydration safely.
 *
 * @returns CodeMirror theme extension (githubLight or githubDark)
 *
 * @example
 * ```tsx
 * const theme = useCodeMirrorTheme()
 *
 * <CodeMirror
 *   value={code}
 *   extensions={[python(), theme]}
 * />
 * ```
 */
export function useCodeMirrorTheme(): Extension {
  const { resolvedTheme } = useTheme()

  const theme = useMemo(() => {
    // resolvedTheme handles 'system' preference and returns actual theme (light/dark)
    return resolvedTheme === 'dark' ? githubDark : githubLight
  }, [resolvedTheme])

  return theme
}
