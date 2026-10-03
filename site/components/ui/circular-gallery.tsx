"use client";

import React, { useState, useEffect, useRef, HTMLAttributes } from "react";

const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(" ");

export interface GalleryItem {
  /** Title shown on the card. */
  title: string;
  /** Optional short caption / subtitle (italic). */
  caption?: string;
  media: {
    /** Image, animated WebP, or GIF URL. */
    url: string;
    /** alt text */
    text: string;
    /** object-position, e.g. "50% 40%" */
    pos?: string;
    /** credit / source */
    by?: string;
  };
}

interface CircularGalleryProps extends HTMLAttributes<HTMLDivElement> {
  items: GalleryItem[];
  /** How far the items sit from the center. */
  radius?: number;
  /** Auto-rotation speed when not scrolling. */
  autoRotateSpeed?: number;
  /** Card size in px. Defaults to widescreen 16:9. */
  cardWidth?: number;
  cardHeight?: number;
}

const CircularGallery = React.forwardRef<HTMLDivElement, CircularGalleryProps>(
  (
    { items, className, radius = 640, autoRotateSpeed = 0.02, cardWidth = 400, cardHeight = 225, ...props },
    ref,
  ) => {
    const [rotation, setRotation] = useState(0);
    const [isScrolling, setIsScrolling] = useState(false);
    const scrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const animationFrameRef = useRef<number | null>(null);

    useEffect(() => {
      const handleScroll = () => {
        setIsScrolling(true);
        if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
        const scrollableHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrollProgress = scrollableHeight > 0 ? window.scrollY / scrollableHeight : 0;
        setRotation(scrollProgress * 360);
        scrollTimeoutRef.current = setTimeout(() => setIsScrolling(false), 150);
      };
      window.addEventListener("scroll", handleScroll, { passive: true });
      return () => {
        window.removeEventListener("scroll", handleScroll);
        if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
      };
    }, []);

    useEffect(() => {
      const autoRotate = () => {
        if (!isScrolling) setRotation((prev) => prev + autoRotateSpeed);
        animationFrameRef.current = requestAnimationFrame(autoRotate);
      };
      animationFrameRef.current = requestAnimationFrame(autoRotate);
      return () => {
        if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      };
    }, [isScrolling, autoRotateSpeed]);

    const anglePerItem = 360 / items.length;

    return (
      <div
        ref={ref}
        role="region"
        aria-label="Circular 3D gallery"
        className={cn("relative flex h-full w-full items-center justify-center", className)}
        style={{ perspective: "2000px" }}
        {...props}
      >
        <div
          className="relative h-full w-full"
          style={{ transform: `rotateY(${rotation}deg)`, transformStyle: "preserve-3d" }}
        >
          {items.map((item, i) => {
            const itemAngle = i * anglePerItem;
            const relativeAngle = (itemAngle + (rotation % 360) + 360) % 360;
            const normalizedAngle = Math.abs(relativeAngle > 180 ? 360 - relativeAngle : relativeAngle);
            const opacity = Math.max(0.3, 1 - normalizedAngle / 180);
            return (
              <div
                key={item.media.url}
                role="group"
                aria-label={item.title}
                className="absolute"
                style={{
                  width: `${cardWidth}px`,
                  height: `${cardHeight}px`,
                  transform: `rotateY(${itemAngle}deg) translateZ(${radius}px)`,
                  left: "50%",
                  top: "50%",
                  marginLeft: `-${cardWidth / 2}px`,
                  marginTop: `-${cardHeight / 2}px`,
                  opacity,
                  transition: "opacity 0.3s linear",
                }}
              >
                <div className="group relative h-full w-full overflow-hidden rounded-xl border border-white/10 bg-white/5 shadow-2xl backdrop-blur-md">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={item.media.url}
                    alt={item.media.text}
                    loading="lazy"
                    decoding="async"
                    className="absolute inset-0 h-full w-full object-cover"
                    style={{ objectPosition: item.media.pos || "center" }}
                  />
                  <div className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-black/85 to-transparent p-4 text-white">
                    <h3 className="text-lg font-medium">{item.title}</h3>
                    {item.caption && <em className="text-sm italic text-white/80">{item.caption}</em>}
                    {item.media.by && <p className="mt-1 text-xs text-white/60">{item.media.by}</p>}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  },
);
CircularGallery.displayName = "CircularGallery";

export { CircularGallery };
