import { ComponentProps } from "react";
import * as ToggleGroupPrimitive from "@radix-ui/react-toggle-group";
import { type VariantProps } from "class-variance-authority";
import { toggleVariants } from "./toggle";
declare function ToggleGroup({ className, variant, size, children, ...props }: ComponentProps<typeof ToggleGroupPrimitive.Root> & VariantProps<typeof toggleVariants>): any;
declare function ToggleGroupItem({ className, children, variant, size, ...props }: ComponentProps<typeof ToggleGroupPrimitive.Item> & VariantProps<typeof toggleVariants>): any;
export { ToggleGroup, ToggleGroupItem };
//# sourceMappingURL=toggle-group.d.ts.map