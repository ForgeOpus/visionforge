import { ComponentProps } from "react";
import * as MenubarPrimitive from "@radix-ui/react-menubar";
declare function Menubar({ className, ...props }: ComponentProps<typeof MenubarPrimitive.Root>): import("react/jsx-runtime").JSX.Element;
declare function MenubarMenu({ ...props }: ComponentProps<typeof MenubarPrimitive.Menu>): import("react/jsx-runtime").JSX.Element;
declare function MenubarGroup({ ...props }: ComponentProps<typeof MenubarPrimitive.Group>): import("react/jsx-runtime").JSX.Element;
declare function MenubarPortal({ ...props }: ComponentProps<typeof MenubarPrimitive.Portal>): import("react/jsx-runtime").JSX.Element;
declare function MenubarRadioGroup({ ...props }: ComponentProps<typeof MenubarPrimitive.RadioGroup>): import("react/jsx-runtime").JSX.Element;
declare function MenubarTrigger({ className, ...props }: ComponentProps<typeof MenubarPrimitive.Trigger>): import("react/jsx-runtime").JSX.Element;
declare function MenubarContent({ className, align, alignOffset, sideOffset, ...props }: ComponentProps<typeof MenubarPrimitive.Content>): import("react/jsx-runtime").JSX.Element;
declare function MenubarItem({ className, inset, variant, ...props }: ComponentProps<typeof MenubarPrimitive.Item> & {
    inset?: boolean;
    variant?: "default" | "destructive";
}): import("react/jsx-runtime").JSX.Element;
declare function MenubarCheckboxItem({ className, children, checked, ...props }: ComponentProps<typeof MenubarPrimitive.CheckboxItem>): import("react/jsx-runtime").JSX.Element;
declare function MenubarRadioItem({ className, children, ...props }: ComponentProps<typeof MenubarPrimitive.RadioItem>): import("react/jsx-runtime").JSX.Element;
declare function MenubarLabel({ className, inset, ...props }: ComponentProps<typeof MenubarPrimitive.Label> & {
    inset?: boolean;
}): import("react/jsx-runtime").JSX.Element;
declare function MenubarSeparator({ className, ...props }: ComponentProps<typeof MenubarPrimitive.Separator>): import("react/jsx-runtime").JSX.Element;
declare function MenubarShortcut({ className, ...props }: ComponentProps<"span">): import("react/jsx-runtime").JSX.Element;
declare function MenubarSub({ ...props }: ComponentProps<typeof MenubarPrimitive.Sub>): import("react/jsx-runtime").JSX.Element;
declare function MenubarSubTrigger({ className, inset, children, ...props }: ComponentProps<typeof MenubarPrimitive.SubTrigger> & {
    inset?: boolean;
}): import("react/jsx-runtime").JSX.Element;
declare function MenubarSubContent({ className, ...props }: ComponentProps<typeof MenubarPrimitive.SubContent>): import("react/jsx-runtime").JSX.Element;
export { Menubar, MenubarPortal, MenubarMenu, MenubarTrigger, MenubarContent, MenubarGroup, MenubarSeparator, MenubarLabel, MenubarItem, MenubarShortcut, MenubarCheckboxItem, MenubarRadioGroup, MenubarRadioItem, MenubarSub, MenubarSubTrigger, MenubarSubContent, };
//# sourceMappingURL=menubar.d.ts.map