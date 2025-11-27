import { ComponentProps } from "react";
import { type VariantProps } from "class-variance-authority";
declare const alertVariants: any;
declare function Alert({ className, variant, ...props }: ComponentProps<"div"> & VariantProps<typeof alertVariants>): any;
declare function AlertTitle({ className, ...props }: ComponentProps<"div">): any;
declare function AlertDescription({ className, ...props }: ComponentProps<"div">): any;
export { Alert, AlertTitle, AlertDescription };
//# sourceMappingURL=alert.d.ts.map