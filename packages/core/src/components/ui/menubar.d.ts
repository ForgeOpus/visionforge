import { ComponentProps } from "react";
import * as MenubarPrimitive from "@radix-ui/react-menubar";
declare function Menubar({ className, ...props }: ComponentProps<typeof MenubarPrimitive.Root>): any;
declare function MenubarMenu({ ...props }: ComponentProps<typeof MenubarPrimitive.Menu>): any;
declare function MenubarGroup({ ...props }: ComponentProps<typeof MenubarPrimitive.Group>): any;
declare function MenubarPortal({ ...props }: ComponentProps<typeof MenubarPrimitive.Portal>): any;
declare function MenubarRadioGroup({ ...props }: ComponentProps<typeof MenubarPrimitive.RadioGroup>): any;
declare function MenubarTrigger({ className, ...props }: ComponentProps<typeof MenubarPrimitive.Trigger>): any;
declare function MenubarContent({ className, align, alignOffset, sideOffset, ...props }: ComponentProps<typeof MenubarPrimitive.Content>): any;
declare function MenubarItem({ className, inset, variant, ...props }: ComponentProps<typeof MenubarPrimitive.Item> & {
    inset?: boolean;
    variant?: "default" | "destructive";
}): any;
declare function MenubarCheckboxItem({ className, children, checked, ...props }: ComponentProps<typeof MenubarPrimitive.CheckboxItem>): any;
declare function MenubarRadioItem({ className, children, ...props }: ComponentProps<typeof MenubarPrimitive.RadioItem>): any;
declare function MenubarLabel({ className, inset, ...props }: ComponentProps<typeof MenubarPrimitive.Label> & {
    inset?: boolean;
}): any;
declare function MenubarSeparator({ className, ...props }: ComponentProps<typeof MenubarPrimitive.Separator>): any;
declare function MenubarShortcut({ className, ...props }: ComponentProps<"span">): any;
declare function MenubarSub({ ...props }: ComponentProps<typeof MenubarPrimitive.Sub>): any;
declare function MenubarSubTrigger({ className, inset, children, ...props }: ComponentProps<typeof MenubarPrimitive.SubTrigger> & {
    inset?: boolean;
}): any;
declare function MenubarSubContent({ className, ...props }: ComponentProps<typeof MenubarPrimitive.SubContent>): any;
export { Menubar, MenubarPortal, MenubarMenu, MenubarTrigger, MenubarContent, MenubarGroup, MenubarSeparator, MenubarLabel, MenubarItem, MenubarShortcut, MenubarCheckboxItem, MenubarRadioGroup, MenubarRadioItem, MenubarSub, MenubarSubTrigger, MenubarSubContent, };
//# sourceMappingURL=menubar.d.ts.map