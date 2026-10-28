"use client";

import { useCallback, useEffect, useState } from "react";

export interface GalleryItem {
  /** Title shown on the tile / lightbox. */
  title: string;
  /** Optional caption. */
  caption?: string;
  media: {
    /** Thumbnail: image, animated WebP, or GIF URL (shown in the grid). */
    url: string;
    /** Optional full video (mp4/webm). If set, the lightbox plays this with controls. */
    video?: string;
    text: string;
    pos?: string;
    by?: string;
  };
}

export function ImageGallery({ items }: { items: GalleryItem[] }) {
  const [open, setOpen] = useState<number | null>(null);

  const close = useCallback(() => setOpen(null), []);
  const go = useCallback(
    (dir: number) => setOpen((i) => (i === null ? i : (i + dir + items.length) % items.length)),
    [items.length],
  );

  useEffect(() => {
    if (open === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
      else if (e.key === "ArrowRight") go(1);
      else if (e.key === "ArrowLeft") go(-1);
    };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open, close, go]);

  const active = open === null ? null : items[open];

  return (
    <>
      {/* expanding strip — each tile grows on hover; click opens it full-screen */}
      <div className="flex h-[300px] w-full gap-2 sm:h-[440px]">
        {items.map((item, i) => (
          <button
            key={item.media.url}
            onClick={() => setOpen(i)}
            aria-label={`Open ${item.title}`}
            className="group relative flex-1 overflow-hidden rounded-lg outline-none transition-all duration-500 ease-out hover:flex-[4] focus-visible:ring-2 focus-visible:ring-white/40"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={item.media.url}
              alt={item.media.text}
              loading="lazy"
              decoding="async"
              className="h-full w-full object-cover object-center"
              style={{ objectPosition: item.media.pos || "center" }}
            />
            <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-3 text-left opacity-0 transition-opacity duration-300 group-hover:opacity-100">
              <h3 className="whitespace-nowrap text-sm font-medium text-white">{item.title}</h3>
              {item.caption && <p className="text-xs text-white/60">{item.caption}</p>}
            </div>
          </button>
        ))}
      </div>

      {/* lightbox */}
      {active && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4 backdrop-blur-sm"
          onClick={close}
          role="dialog"
          aria-modal="true"
          aria-label={active.title}
        >
          <button
            onClick={close}
            aria-label="Close"
            className="absolute right-4 top-4 flex h-10 w-10 items-center justify-center rounded-full text-2xl text-white/70 hover:bg-white/10 hover:text-white"
          >
            ✕
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); go(-1); }}
            aria-label="Previous"
            className="absolute left-3 flex h-12 w-12 items-center justify-center rounded-full text-3xl text-white/70 hover:bg-white/10 hover:text-white sm:left-6"
          >
            ‹
          </button>
          <figure className="flex max-h-full max-w-5xl flex-col items-center" onClick={(e) => e.stopPropagation()}>
            {active.media.video ? (
              <video
                src={active.media.video}
                controls
                autoPlay
                loop
                playsInline
                className="max-h-[80vh] w-auto max-w-full rounded-lg"
              />
            ) : (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={active.media.url}
                alt={active.media.text}
                className="max-h-[80vh] w-auto max-w-full rounded-lg object-contain"
              />
            )}
            <figcaption className="mt-3 text-center">
              <span className="text-base font-medium text-white">{active.title}</span>
              {active.media.by && <span className="ml-2 text-sm text-white/50">· {active.media.by}</span>}
            </figcaption>
          </figure>
          <button
            onClick={(e) => { e.stopPropagation(); go(1); }}
            aria-label="Next"
            className="absolute right-3 flex h-12 w-12 items-center justify-center rounded-full text-3xl text-white/70 hover:bg-white/10 hover:text-white sm:right-6"
          >
            ›
          </button>
        </div>
      )}
    </>
  );
}
