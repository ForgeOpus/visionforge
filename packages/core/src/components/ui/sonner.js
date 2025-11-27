import { jsx as _jsx } from "react/jsx-runtime";
import { Toaster as Sonner } from "sonner";
const Toaster = ({ ...props }) => {
    return (_jsx(Sonner, { theme: "system", className: "toaster group", style: {
            "--normal-bg": "var(--popover)",
            "--normal-text": "var(--popover-foreground)",
            "--normal-border": "var(--border)",
        }, ...props }));
};
export { Toaster };
