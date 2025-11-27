import { ComponentProps } from "react";
import { Drawer as DrawerPrimitive } from "vaul";
declare function Drawer({ ...props }: ComponentProps<typeof DrawerPrimitive.Root>): any;
declare function DrawerTrigger({ ...props }: ComponentProps<typeof DrawerPrimitive.Trigger>): any;
declare function DrawerPortal({ ...props }: ComponentProps<typeof DrawerPrimitive.Portal>): any;
declare function DrawerClose({ ...props }: ComponentProps<typeof DrawerPrimitive.Close>): any;
declare function DrawerOverlay({ className, ...props }: ComponentProps<typeof DrawerPrimitive.Overlay>): any;
declare function DrawerContent({ className, children, ...props }: ComponentProps<typeof DrawerPrimitive.Content>): any;
declare function DrawerHeader({ className, ...props }: ComponentProps<"div">): any;
declare function DrawerFooter({ className, ...props }: ComponentProps<"div">): any;
declare function DrawerTitle({ className, ...props }: ComponentProps<typeof DrawerPrimitive.Title>): any;
declare function DrawerDescription({ className, ...props }: ComponentProps<typeof DrawerPrimitive.Description>): any;
export { Drawer, DrawerPortal, DrawerOverlay, DrawerTrigger, DrawerClose, DrawerContent, DrawerHeader, DrawerFooter, DrawerTitle, DrawerDescription, };
//# sourceMappingURL=drawer.d.ts.map