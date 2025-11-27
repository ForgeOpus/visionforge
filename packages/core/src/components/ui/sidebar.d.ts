import { ComponentProps } from "react";
import { VariantProps } from "class-variance-authority";
import { Button } from "./button";
import { Input } from "./input";
import { Separator } from "./separator";
import { TooltipContent } from "./tooltip";
type SidebarContextProps = {
    state: "expanded" | "collapsed";
    open: boolean;
    setOpen: (open: boolean) => void;
    openMobile: boolean;
    setOpenMobile: (open: boolean) => void;
    isMobile: boolean;
    toggleSidebar: () => void;
};
declare function useSidebar(): SidebarContextProps;
declare function SidebarProvider({ defaultOpen, open: openProp, onOpenChange: setOpenProp, className, style, children, ...props }: ComponentProps<"div"> & {
    defaultOpen?: boolean;
    open?: boolean;
    onOpenChange?: (open: boolean) => void;
}): import("react/jsx-runtime").JSX.Element;
declare function Sidebar({ side, variant, collapsible, className, children, ...props }: ComponentProps<"div"> & {
    side?: "left" | "right";
    variant?: "sidebar" | "floating" | "inset";
    collapsible?: "offcanvas" | "icon" | "none";
}): import("react/jsx-runtime").JSX.Element;
declare function SidebarTrigger({ className, onClick, ...props }: ComponentProps<typeof Button>): import("react/jsx-runtime").JSX.Element;
declare function SidebarRail({ className, ...props }: ComponentProps<"button">): import("react/jsx-runtime").JSX.Element;
declare function SidebarInset({ className, ...props }: ComponentProps<"main">): import("react/jsx-runtime").JSX.Element;
declare function SidebarInput({ className, ...props }: ComponentProps<typeof Input>): import("react/jsx-runtime").JSX.Element;
declare function SidebarHeader({ className, ...props }: ComponentProps<"div">): import("react/jsx-runtime").JSX.Element;
declare function SidebarFooter({ className, ...props }: ComponentProps<"div">): import("react/jsx-runtime").JSX.Element;
declare function SidebarSeparator({ className, ...props }: ComponentProps<typeof Separator>): import("react/jsx-runtime").JSX.Element;
declare function SidebarContent({ className, ...props }: ComponentProps<"div">): import("react/jsx-runtime").JSX.Element;
declare function SidebarGroup({ className, ...props }: ComponentProps<"div">): import("react/jsx-runtime").JSX.Element;
declare function SidebarGroupLabel({ className, asChild, ...props }: ComponentProps<"div"> & {
    asChild?: boolean;
}): import("react/jsx-runtime").JSX.Element;
declare function SidebarGroupAction({ className, asChild, ...props }: ComponentProps<"button"> & {
    asChild?: boolean;
}): import("react/jsx-runtime").JSX.Element;
declare function SidebarGroupContent({ className, ...props }: ComponentProps<"div">): import("react/jsx-runtime").JSX.Element;
declare function SidebarMenu({ className, ...props }: ComponentProps<"ul">): import("react/jsx-runtime").JSX.Element;
declare function SidebarMenuItem({ className, ...props }: ComponentProps<"li">): import("react/jsx-runtime").JSX.Element;
declare const sidebarMenuButtonVariants: (props?: ({
    variant?: "default" | "outline" | null | undefined;
    size?: "default" | "sm" | "lg" | null | undefined;
} & import("class-variance-authority/types").ClassProp) | undefined) => string;
declare function SidebarMenuButton({ asChild, isActive, variant, size, tooltip, className, ...props }: ComponentProps<"button"> & {
    asChild?: boolean;
    isActive?: boolean;
    tooltip?: string | ComponentProps<typeof TooltipContent>;
} & VariantProps<typeof sidebarMenuButtonVariants>): import("react/jsx-runtime").JSX.Element;
declare function SidebarMenuAction({ className, asChild, showOnHover, ...props }: ComponentProps<"button"> & {
    asChild?: boolean;
    showOnHover?: boolean;
}): import("react/jsx-runtime").JSX.Element;
declare function SidebarMenuBadge({ className, ...props }: ComponentProps<"div">): import("react/jsx-runtime").JSX.Element;
declare function SidebarMenuSkeleton({ className, showIcon, ...props }: ComponentProps<"div"> & {
    showIcon?: boolean;
}): import("react/jsx-runtime").JSX.Element;
declare function SidebarMenuSub({ className, ...props }: ComponentProps<"ul">): import("react/jsx-runtime").JSX.Element;
declare function SidebarMenuSubItem({ className, ...props }: ComponentProps<"li">): import("react/jsx-runtime").JSX.Element;
declare function SidebarMenuSubButton({ asChild, size, isActive, className, ...props }: ComponentProps<"a"> & {
    asChild?: boolean;
    size?: "sm" | "md";
    isActive?: boolean;
}): import("react/jsx-runtime").JSX.Element;
export { Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupAction, SidebarGroupContent, SidebarGroupLabel, SidebarHeader, SidebarInput, SidebarInset, SidebarMenu, SidebarMenuAction, SidebarMenuBadge, SidebarMenuButton, SidebarMenuItem, SidebarMenuSkeleton, SidebarMenuSub, SidebarMenuSubButton, SidebarMenuSubItem, SidebarProvider, SidebarRail, SidebarSeparator, SidebarTrigger, useSidebar, };
//# sourceMappingURL=sidebar.d.ts.map