import { ComponentProps } from "react";
import { type VariantProps } from "class-variance-authority";
declare const badgeVariants: any;
declare function Badge({ className, variant, asChild, ...props }: ComponentProps<"span"> & VariantProps<typeof badgeVariants> & {
    asChild?: boolean;
}): any;
export { Badge, badgeVariants };
//# sourceMappingURL=badge.d.ts.map