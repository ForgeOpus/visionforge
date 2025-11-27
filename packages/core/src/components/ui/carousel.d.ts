import { ComponentProps } from "react";
import useEmblaCarousel, { type UseEmblaCarouselType } from "embla-carousel-react";
import { Button } from "./button";
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
declare function Carousel({ orientation, opts, setApi, plugins, className, children, ...props }: ComponentProps<"div"> & CarouselProps): any;
declare function CarouselContent({ className, ...props }: ComponentProps<"div">): any;
declare function CarouselItem({ className, ...props }: ComponentProps<"div">): any;
declare function CarouselPrevious({ className, variant, size, ...props }: ComponentProps<typeof Button>): any;
declare function CarouselNext({ className, variant, size, ...props }: ComponentProps<typeof Button>): any;
export { type CarouselApi, Carousel, CarouselContent, CarouselItem, CarouselPrevious, CarouselNext, };
//# sourceMappingURL=carousel.d.ts.map