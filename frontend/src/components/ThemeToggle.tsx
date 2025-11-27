import { useEffect, useState } from "react";
import { Sun, Moon } from "@phosphor-icons/react";
import { Button } from "@visionforge/core/components/ui/button";

export function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Avoid hydration mismatch by only rendering after mount
  useEffect(() => {
    setMounted(true);
    // Check if dark mode is preferred
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setIsDark(prefersDark);
  }, []);

  if (!mounted) {
    return (
      <div className="h-9 w-9 rounded-md flex items-center justify-center bg-muted animate-pulse" />
    );
  }

  const handleToggle = () => {
    setIsDark(!isDark);
    // Note: This is a placeholder - proper theme switching would require additional setup
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={handleToggle}
      className="h-9 w-9 rounded-md relative hover:bg-accent hover:text-accent-foreground transition-colors"
      aria-label="Toggle theme"
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {isDark ? (
        <Moon
          size={20}
          weight="fill"
          className="text-foreground transition-transform duration-200 hover:scale-110"
          aria-hidden="true"
        />
      ) : (
        <Sun
          size={20}
          weight="fill"
          className="text-foreground transition-transform duration-200 hover:scale-110"
          aria-hidden="true"
        />
      )}
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
}
