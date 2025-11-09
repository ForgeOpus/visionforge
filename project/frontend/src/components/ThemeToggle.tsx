import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Classic } from "@theme-toggles/react";
import "@theme-toggles/react/css/Classic.css";

export function ThemeToggle() {
  const { setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Avoid hydration mismatch by only rendering after mount
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="h-9 w-9 rounded-md flex items-center justify-center">
        <Classic duration={750} />
      </div>
    );
  }

  // Use resolvedTheme to handle system preference
  const isDark = resolvedTheme === "dark";

  const handleToggle = () => {
    const newTheme = isDark ? "light" : "dark";
    setTheme(newTheme);
  };

  return (
    <div className="theme-toggle-wrapper h-9 w-9 rounded-md flex items-center justify-center hover:bg-muted hover:scale-110 transition-all cursor-pointer">
      <Classic
        duration={750}
        toggled={isDark}
        toggle={handleToggle}
        className="text-foreground"
      />
    </div>
  );
}
