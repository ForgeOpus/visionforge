import { useEffect, useState } from "react";
import { Sun, Moon } from "@phosphor-icons/react";
import { Button } from "@visionforge/core/components/ui/button";

export function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Load theme from localStorage on mount
  useEffect(() => {
    setMounted(true);

    // Check localStorage first, then system preference
    const savedTheme = localStorage.getItem("theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const shouldBeDark = savedTheme === "dark" || (!savedTheme && prefersDark);

    setIsDark(shouldBeDark);

    // Apply theme to document
    if (shouldBeDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, []);

  if (!mounted) {
    return (
      <div className="h-9 w-9 rounded-md flex items-center justify-center bg-muted animate-pulse" />
    );
  }

  const handleToggle = () => {
    const newIsDark = !isDark;
    setIsDark(newIsDark);

    // Apply theme to document
    if (newIsDark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
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
