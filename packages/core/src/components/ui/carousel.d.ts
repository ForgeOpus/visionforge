import { ComponentProps } from "react";
import useEmblaCarousel, { type UseEmblaCarouselType } from "embla-carousel-react";
import { Button } from "./ui/button";
type CarouselApi = UseEmblaCarouselType[1];
type UseCarouselParameters = Parameters<typeof useEmblaCarousel>;
type CarouselOptions = UseCarouselParameters[0];
type CarouselPlugin = UseCarouselParameters[1];
type CarouselProps = {
    opts?: CarouselOptions;
    plugins?: CarouselPlugin;
    orientation?: "horizontal" | "vertical";
    setApi?: (api: CarouselApi) => void;
};
declare function Carousel({ orientation, opts, setApi, plugins, className, children, ...props }: ComponentProps<"div"> & CarouselProps): import("react/jsx-runtime").JSX.Element;
declare function CarouselContent({ className, ...props }: ComponentProps<"div">): import("react/jsx-runtime").JSX.Element;
declare function CarouselItem({ className, ...props }: ComponentProps<"div">): import("react/jsx-runtime").JSX.Element;
declare function CarouselPrevious({ className, variant, size, ...props }: ComponentProps<typeof Button>): import("react/jsx-runtime").JSX.Element;
declare function CarouselNext({ className, variant, size, ...props }: ComponentProps<typeof Button>): import("react/jsx-runtime").JSX.Element;
export { type CarouselApi, Carousel, CarouselContent, CarouselItem, CarouselPrevious, CarouselNext, };
//# sourceMappingURL=carousel.d.ts.map