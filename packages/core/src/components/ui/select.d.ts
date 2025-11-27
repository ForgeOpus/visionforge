import { ComponentProps } from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
declare function Select({ ...props }: ComponentProps<typeof SelectPrimitive.Root>): any;
declare function SelectGroup({ ...props }: ComponentProps<typeof SelectPrimitive.Group>): any;
declare function SelectValue({ ...props }: ComponentProps<typeof SelectPrimitive.Value>): any;
declare function SelectTrigger({ className, size, children, ...props }: ComponentProps<typeof SelectPrimitive.Trigger> & {
    size?: "sm" | "default";
}): any;
declare function SelectContent({ className, children, position, ...props }: ComponentProps<typeof SelectPrimitive.Content>): any;
declare function SelectLabel({ className, ...props }: ComponentProps<typeof SelectPrimitive.Label>): any;
declare function SelectItem({ className, children, ...props }: ComponentProps<typeof SelectPrimitive.Item>): any;
declare function SelectSeparator({ className, ...props }: ComponentProps<typeof SelectPrimitive.Separator>): any;
declare function SelectScrollUpButton({ className, ...props }: ComponentProps<typeof SelectPrimitive.ScrollUpButton>): any;
declare function SelectScrollDownButton({ className, ...props }: ComponentProps<typeof SelectPrimitive.ScrollDownButton>): any;
export { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectScrollDownButton, SelectScrollUpButton, SelectSeparator, SelectTrigger, SelectValue, };
//# sourceMappingURL=select.d.ts.map