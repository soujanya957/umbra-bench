import { ImageGallery, type GalleryItem } from "@/components/ui/image-gallery";

// Prefix local /public assets with the base path (needed on GitHub Pages).
// External URLs (http...) are returned unchanged.
const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const asset = (p: string) => (p.startsWith("http") ? p : base + p);

// Demo clips in `public/demos/`. Add more by dropping a file there and adding a
// line below. Animated WebP/GIF work because each tile renders an <img>.
const demos: GalleryItem[] = [
  { title: "Bloom", media: { url: asset("/demos/bloom.webp"), text: "bloom", by: "UMBRA" } },
  { title: "Bloom robot", media: { url: asset("/demos/bloom_robot.webp"), text: "bloom robot", by: "UMBRA" } },
  { title: "Cat", media: { url: asset("/demos/cat.webp"), text: "cat", by: "UMBRA" } },
  { title: "Crocodile", media: { url: asset("/demos/crocodile.webp"), text: "crocodile", by: "UMBRA" } },
  { title: "Howling at the moon", media: { url: asset("/demos/howling_at_the_moon.webp"), text: "howling at the moon", by: "UMBRA" } },
  { title: "Spinning star", media: { url: asset("/demos/spinning_star.webp"), text: "spinning star", by: "UMBRA" } },
];

export default function DemoGallery() {
  return (
    <section id="video" className="mx-auto max-w-6xl px-6 py-24">
      <div className="mb-10 text-center">
        <h2 className="text-xs font-semibold uppercase tracking-[0.3em] text-umbra-glow/70">Demos</h2>
      </div>
      <ImageGallery items={demos} />
    </section>
  );
}
