import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Sun, Moon } from "@phosphor-icons/react";
import { Button } from "@/components/ui/button";

export function ThemeToggle() {
  const { setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Avoid hydration mismatch by only rendering after mount
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="h-9 w-9 rounded-md flex items-center justify-center bg-muted animate-pulse" />
    );
  }

  // Use resolvedTheme to handle system preference
  const isDark = resolvedTheme === "dark";

  const handleToggle = () => {
    const newTheme = isDark ? "light" : "dark";
    setTheme(newTheme);
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={handleToggle}
      className="h-9 w-9 rounded-md relative"
      aria-label="Toggle theme"
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      <Sun
        size={20}
        weight="fill"
        className="rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0 text-foreground"
        aria-hidden="true"
      />
      <Moon
        size={20}
        weight="fill"
        className="absolute rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100 text-foreground"
        aria-hidden="true"
      />
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
}
