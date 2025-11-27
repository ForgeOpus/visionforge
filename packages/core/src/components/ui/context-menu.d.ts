import { ComponentProps } from "react";
import * as ContextMenuPrimitive from "@radix-ui/react-context-menu";
declare function ContextMenu({ ...props }: ComponentProps<typeof ContextMenuPrimitive.Root>): any;
declare function ContextMenuTrigger({ ...props }: ComponentProps<typeof ContextMenuPrimitive.Trigger>): any;
declare function ContextMenuGroup({ ...props }: ComponentProps<typeof ContextMenuPrimitive.Group>): any;
declare function ContextMenuPortal({ ...props }: ComponentProps<typeof ContextMenuPrimitive.Portal>): any;
declare function ContextMenuSub({ ...props }: ComponentProps<typeof ContextMenuPrimitive.Sub>): any;
declare function ContextMenuRadioGroup({ ...props }: ComponentProps<typeof ContextMenuPrimitive.RadioGroup>): any;
declare function ContextMenuSubTrigger({ className, inset, children, ...props }: ComponentProps<typeof ContextMenuPrimitive.SubTrigger> & {
    inset?: boolean;
}): any;
declare function ContextMenuSubContent({ className, ...props }: ComponentProps<typeof ContextMenuPrimitive.SubContent>): any;
declare function ContextMenuContent({ className, ...props }: ComponentProps<typeof ContextMenuPrimitive.Content>): any;
declare function ContextMenuItem({ className, inset, variant, ...props }: ComponentProps<typeof ContextMenuPrimitive.Item> & {
    inset?: boolean;
    variant?: "default" | "destructive";
}): any;
declare function ContextMenuCheckboxItem({ className, children, checked, ...props }: ComponentProps<typeof ContextMenuPrimitive.CheckboxItem>): any;
declare function ContextMenuRadioItem({ className, children, ...props }: ComponentProps<typeof ContextMenuPrimitive.RadioItem>): any;
declare function ContextMenuLabel({ className, inset, ...props }: ComponentProps<typeof ContextMenuPrimitive.Label> & {
    inset?: boolean;
}): any;
declare function ContextMenuSeparator({ className, ...props }: ComponentProps<typeof ContextMenuPrimitive.Separator>): any;
declare function ContextMenuShortcut({ className, ...props }: ComponentProps<"span">): any;
export { ContextMenu, ContextMenuTrigger, ContextMenuContent, ContextMenuItem, ContextMenuCheckboxItem, ContextMenuRadioItem, ContextMenuLabel, ContextMenuSeparator, ContextMenuShortcut, ContextMenuGroup, ContextMenuPortal, ContextMenuSub, ContextMenuSubContent, ContextMenuSubTrigger, ContextMenuRadioGroup, };
//# sourceMappingURL=context-menu.d.ts.map