import { ComponentProps } from "react";
import * as SheetPrimitive from "@radix-ui/react-dialog";
declare function Sheet({ ...props }: ComponentProps<typeof SheetPrimitive.Root>): any;
declare function SheetTrigger({ ...props }: ComponentProps<typeof SheetPrimitive.Trigger>): any;
declare function SheetClose({ ...props }: ComponentProps<typeof SheetPrimitive.Close>): any;
declare function SheetContent({ className, children, side, ...props }: ComponentProps<typeof SheetPrimitive.Content> & {
    side?: "top" | "right" | "bottom" | "left";
}): any;
declare function SheetHeader({ className, ...props }: ComponentProps<"div">): any;
declare function SheetFooter({ className, ...props }: ComponentProps<"div">): any;
declare function SheetTitle({ className, ...props }: ComponentProps<typeof SheetPrimitive.Title>): any;
declare function SheetDescription({ className, ...props }: ComponentProps<typeof SheetPrimitive.Description>): any;
export { Sheet, SheetTrigger, SheetClose, SheetContent, SheetHeader, SheetFooter, SheetTitle, SheetDescription, };
//# sourceMappingURL=sheet.d.ts.map