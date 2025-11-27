import { ComponentProps } from "react";
import { Button } from "./button";
declare function Pagination({ className, ...props }: ComponentProps<"nav">): import("react/jsx-runtime").JSX.Element;
declare function PaginationContent({ className, ...props }: ComponentProps<"ul">): import("react/jsx-runtime").JSX.Element;
declare function PaginationItem({ ...props }: ComponentProps<"li">): import("react/jsx-runtime").JSX.Element;
type PaginationLinkProps = {
    isActive?: boolean;
} & Pick<ComponentProps<typeof Button>, "size"> & ComponentProps<"a">;
declare function PaginationLink({ className, isActive, size, ...props }: PaginationLinkProps): import("react/jsx-runtime").JSX.Element;
declare function PaginationPrevious({ className, ...props }: ComponentProps<typeof PaginationLink>): import("react/jsx-runtime").JSX.Element;
declare function PaginationNext({ className, ...props }: ComponentProps<typeof PaginationLink>): import("react/jsx-runtime").JSX.Element;
declare function PaginationEllipsis({ className, ...props }: ComponentProps<"span">): import("react/jsx-runtime").JSX.Element;
export { Pagination, PaginationContent, PaginationLink, PaginationItem, PaginationPrevious, PaginationNext, PaginationEllipsis, };
//# sourceMappingURL=pagination.d.ts.map