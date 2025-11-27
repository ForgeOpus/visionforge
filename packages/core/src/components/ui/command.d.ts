import { ComponentProps } from "react";
import { Command as CommandPrimitive } from "cmdk";
import { Dialog } from "./ui/dialog";
declare function Command({ className, ...props }: ComponentProps<typeof CommandPrimitive>): any;
declare function CommandDialog({ title, description, children, ...props }: ComponentProps<typeof Dialog> & {
    title?: string;
    description?: string;
}): any;
declare function CommandInput({ className, ...props }: ComponentProps<typeof CommandPrimitive.Input>): any;
declare function CommandList({ className, ...props }: ComponentProps<typeof CommandPrimitive.List>): any;
declare function CommandEmpty({ ...props }: ComponentProps<typeof CommandPrimitive.Empty>): any;
declare function CommandGroup({ className, ...props }: ComponentProps<typeof CommandPrimitive.Group>): any;
declare function CommandSeparator({ className, ...props }: ComponentProps<typeof CommandPrimitive.Separator>): any;
declare function CommandItem({ className, ...props }: ComponentProps<typeof CommandPrimitive.Item>): any;
declare function CommandShortcut({ className, ...props }: ComponentProps<"span">): any;
export { Command, CommandDialog, CommandInput, CommandList, CommandEmpty, CommandGroup, CommandItem, CommandShortcut, CommandSeparator, };
//# sourceMappingURL=command.d.ts.map