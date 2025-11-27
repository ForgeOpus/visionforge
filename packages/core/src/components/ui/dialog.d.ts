import { ComponentProps } from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
declare function Dialog({ ...props }: ComponentProps<typeof DialogPrimitive.Root>): any;
declare function DialogTrigger({ ...props }: ComponentProps<typeof DialogPrimitive.Trigger>): any;
declare function DialogPortal({ ...props }: ComponentProps<typeof DialogPrimitive.Portal>): any;
declare function DialogClose({ ...props }: ComponentProps<typeof DialogPrimitive.Close>): any;
declare function DialogOverlay({ className, ...props }: ComponentProps<typeof DialogPrimitive.Overlay>): any;
declare function DialogContent({ className, children, ...props }: ComponentProps<typeof DialogPrimitive.Content>): any;
declare function DialogHeader({ className, ...props }: ComponentProps<"div">): any;
declare function DialogFooter({ className, ...props }: ComponentProps<"div">): any;
declare function DialogTitle({ className, ...props }: ComponentProps<typeof DialogPrimitive.Title>): any;
declare function DialogDescription({ className, ...props }: ComponentProps<typeof DialogPrimitive.Description>): any;
export { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogOverlay, DialogPortal, DialogTitle, DialogTrigger, };
//# sourceMappingURL=dialog.d.ts.map