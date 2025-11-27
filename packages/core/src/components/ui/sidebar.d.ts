import { ComponentProps } from "react";
import { VariantProps } from "class-variance-authority";
import { Button } from "./button";
import { Input } from "./input";
import { Separator } from "./separator";
import { TooltipContent } from "./tooltip";
declare function useSidebar(): any;
declare function SidebarProvider({ defaultOpen, open: openProp, onOpenChange: setOpenProp, className, style, children, ...props }: ComponentProps<"div"> & {
    defaultOpen?: boolean;
    open?: boolean;
    onOpenChange?: (open: boolean) => void;
}): any;
declare function Sidebar({ side, variant, collapsible, className, children, ...props }: ComponentProps<"div"> & {
    side?: "left" | "right";
    variant?: "sidebar" | "floating" | "inset";
    collapsible?: "offcanvas" | "icon" | "none";
}): any;
declare function SidebarTrigger({ className, onClick, ...props }: ComponentProps<typeof Button>): any;
declare function SidebarRail({ className, ...props }: ComponentProps<"button">): any;
declare function SidebarInset({ className, ...props }: ComponentProps<"main">): any;
declare function SidebarInput({ className, ...props }: ComponentProps<typeof Input>): any;
declare function SidebarHeader({ className, ...props }: ComponentProps<"div">): any;
declare function SidebarFooter({ className, ...props }: ComponentProps<"div">): any;
declare function SidebarSeparator({ className, ...props }: ComponentProps<typeof Separator>): any;
declare function SidebarContent({ className, ...props }: ComponentProps<"div">): any;
declare function SidebarGroup({ className, ...props }: ComponentProps<"div">): any;
declare function SidebarGroupLabel({ className, asChild, ...props }: ComponentProps<"div"> & {
    asChild?: boolean;
}): any;
declare function SidebarGroupAction({ className, asChild, ...props }: ComponentProps<"button"> & {
    asChild?: boolean;
}): any;
declare function SidebarGroupContent({ className, ...props }: ComponentProps<"div">): any;
declare function SidebarMenu({ className, ...props }: ComponentProps<"ul">): any;
declare function SidebarMenuItem({ className, ...props }: ComponentProps<"li">): any;
declare const sidebarMenuButtonVariants: any;
declare function SidebarMenuButton({ asChild, isActive, variant, size, tooltip, className, ...props }: ComponentProps<"button"> & {
    asChild?: boolean;
    isActive?: boolean;
    tooltip?: string | ComponentProps<typeof TooltipContent>;
} & VariantProps<typeof sidebarMenuButtonVariants>): any;
declare function SidebarMenuAction({ className, asChild, showOnHover, ...props }: ComponentProps<"button"> & {
    asChild?: boolean;
    showOnHover?: boolean;
}): any;
declare function SidebarMenuBadge({ className, ...props }: ComponentProps<"div">): any;
declare function SidebarMenuSkeleton({ className, showIcon, ...props }: ComponentProps<"div"> & {
    showIcon?: boolean;
}): any;
declare function SidebarMenuSub({ className, ...props }: ComponentProps<"ul">): any;
declare function SidebarMenuSubItem({ className, ...props }: ComponentProps<"li">): any;
declare function SidebarMenuSubButton({ asChild, size, isActive, className, ...props }: ComponentProps<"a"> & {
    asChild?: boolean;
    size?: "sm" | "md";
    isActive?: boolean;
}): any;
export { Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupAction, SidebarGroupContent, SidebarGroupLabel, SidebarHeader, SidebarInput, SidebarInset, SidebarMenu, SidebarMenuAction, SidebarMenuBadge, SidebarMenuButton, SidebarMenuItem, SidebarMenuSkeleton, SidebarMenuSub, SidebarMenuSubButton, SidebarMenuSubItem, SidebarProvider, SidebarRail, SidebarSeparator, SidebarTrigger, useSidebar, };
//# sourceMappingURL=sidebar.d.ts.map