import { ComponentProps } from "react";
import { type VariantProps } from "class-variance-authority";
declare const buttonVariants: any;
declare function Button({ className, variant, size, asChild, ...props }: ComponentProps<"button"> & VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
}): any;
export { Button, buttonVariants };
//# sourceMappingURL=button.d.ts.map