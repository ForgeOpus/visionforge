import { ComponentProps } from "react";
import { Button } from "./button";
declare function Pagination({ className, ...props }: ComponentProps<"nav">): any;
declare function PaginationContent({ className, ...props }: ComponentProps<"ul">): any;
declare function PaginationItem({ ...props }: ComponentProps<"li">): any;
type PaginationLinkProps = {
    isActive?: boolean;
} & Pick<ComponentProps<typeof Button>, "size"> & ComponentProps<"a">;
declare function PaginationLink({ className, isActive, size, ...props }: PaginationLinkProps): any;
declare function PaginationPrevious({ className, ...props }: ComponentProps<typeof PaginationLink>): any;
declare function PaginationNext({ className, ...props }: ComponentProps<typeof PaginationLink>): any;
declare function PaginationEllipsis({ className, ...props }: ComponentProps<"span">): any;
export { Pagination, PaginationContent, PaginationLink, PaginationItem, PaginationPrevious, PaginationNext, PaginationEllipsis, };
//# sourceMappingURL=pagination.d.ts.map