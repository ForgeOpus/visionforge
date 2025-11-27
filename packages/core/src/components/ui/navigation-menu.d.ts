import { ComponentProps } from "react";
import * as NavigationMenuPrimitive from "@radix-ui/react-navigation-menu";
declare function NavigationMenu({ className, children, viewport, ...props }: ComponentProps<typeof NavigationMenuPrimitive.Root> & {
    viewport?: boolean;
}): any;
declare function NavigationMenuList({ className, ...props }: ComponentProps<typeof NavigationMenuPrimitive.List>): any;
declare function NavigationMenuItem({ className, ...props }: ComponentProps<typeof NavigationMenuPrimitive.Item>): any;
declare const navigationMenuTriggerStyle: any;
declare function NavigationMenuTrigger({ className, children, ...props }: ComponentProps<typeof NavigationMenuPrimitive.Trigger>): any;
declare function NavigationMenuContent({ className, ...props }: ComponentProps<typeof NavigationMenuPrimitive.Content>): any;
declare function NavigationMenuViewport({ className, ...props }: ComponentProps<typeof NavigationMenuPrimitive.Viewport>): any;
declare function NavigationMenuLink({ className, ...props }: ComponentProps<typeof NavigationMenuPrimitive.Link>): any;
declare function NavigationMenuIndicator({ className, ...props }: ComponentProps<typeof NavigationMenuPrimitive.Indicator>): any;
export { NavigationMenu, NavigationMenuList, NavigationMenuItem, NavigationMenuContent, NavigationMenuTrigger, NavigationMenuLink, NavigationMenuIndicator, NavigationMenuViewport, navigationMenuTriggerStyle, };
//# sourceMappingURL=navigation-menu.d.ts.map