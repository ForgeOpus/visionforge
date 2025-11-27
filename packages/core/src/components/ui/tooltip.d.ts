import { ComponentProps } from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
declare function TooltipProvider({ delayDuration, ...props }: ComponentProps<typeof TooltipPrimitive.Provider>): any;
declare function Tooltip({ ...props }: ComponentProps<typeof TooltipPrimitive.Root>): any;
declare function TooltipTrigger({ ...props }: ComponentProps<typeof TooltipPrimitive.Trigger>): any;
declare function TooltipContent({ className, sideOffset, children, ...props }: ComponentProps<typeof TooltipPrimitive.Content>): any;
export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider };
//# sourceMappingURL=tooltip.d.ts.map