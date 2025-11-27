import { ComponentProps } from "react";
import * as DropdownMenuPrimitive from "@radix-ui/react-dropdown-menu";
declare function DropdownMenu({ ...props }: ComponentProps<typeof DropdownMenuPrimitive.Root>): any;
declare function DropdownMenuPortal({ ...props }: ComponentProps<typeof DropdownMenuPrimitive.Portal>): any;
declare function DropdownMenuTrigger({ ...props }: ComponentProps<typeof DropdownMenuPrimitive.Trigger>): any;
declare function DropdownMenuContent({ className, sideOffset, ...props }: ComponentProps<typeof DropdownMenuPrimitive.Content>): any;
declare function DropdownMenuGroup({ ...props }: ComponentProps<typeof DropdownMenuPrimitive.Group>): any;
declare function DropdownMenuItem({ className, inset, variant, ...props }: ComponentProps<typeof DropdownMenuPrimitive.Item> & {
    inset?: boolean;
    variant?: "default" | "destructive";
}): any;
declare function DropdownMenuCheckboxItem({ className, children, checked, ...props }: ComponentProps<typeof DropdownMenuPrimitive.CheckboxItem>): any;
declare function DropdownMenuRadioGroup({ ...props }: ComponentProps<typeof DropdownMenuPrimitive.RadioGroup>): any;
declare function DropdownMenuRadioItem({ className, children, ...props }: ComponentProps<typeof DropdownMenuPrimitive.RadioItem>): any;
declare function DropdownMenuLabel({ className, inset, ...props }: ComponentProps<typeof DropdownMenuPrimitive.Label> & {
    inset?: boolean;
}): any;
declare function DropdownMenuSeparator({ className, ...props }: ComponentProps<typeof DropdownMenuPrimitive.Separator>): any;
declare function DropdownMenuShortcut({ className, ...props }: ComponentProps<"span">): any;
declare function DropdownMenuSub({ ...props }: ComponentProps<typeof DropdownMenuPrimitive.Sub>): any;
declare function DropdownMenuSubTrigger({ className, inset, children, ...props }: ComponentProps<typeof DropdownMenuPrimitive.SubTrigger> & {
    inset?: boolean;
}): any;
declare function DropdownMenuSubContent({ className, ...props }: ComponentProps<typeof DropdownMenuPrimitive.SubContent>): any;
export { DropdownMenu, DropdownMenuPortal, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuGroup, DropdownMenuLabel, DropdownMenuItem, DropdownMenuCheckboxItem, DropdownMenuRadioGroup, DropdownMenuRadioItem, DropdownMenuSeparator, DropdownMenuShortcut, DropdownMenuSub, DropdownMenuSubTrigger, DropdownMenuSubContent, };
//# sourceMappingURL=dropdown-menu.d.ts.map