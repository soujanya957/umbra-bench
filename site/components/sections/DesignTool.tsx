const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const asset = (p: string) => (p.startsWith("http") ? p : base + p);

const shots = [
  {
    src: asset("/design/storyboard.png"),
    alt: "Storyboard editor: composing keyframes from a shape library",
    title: "Storyboard",
    caption: "Compose keyframes on a canvas — drop in shapes and 3D cutouts, then scale, rotate, and morph them into each target silhouette.",
  },
  {
    src: asset("/design/casting.png"),
    alt: "Casting view: assigning arms and previewing the shadow projection",
    title: "Casting",
    caption: "Assign each shape to a robot arm and preview the live shadow projection in 3D — arm layout, mount, and light are all adjustable.",
  },
];

export default function DesignTool() {
  return (
    <section id="design" className="mx-auto max-w-5xl px-6 py-24">
      <div className="mb-10 text-center">
        <h2 className="text-xs font-semibold uppercase tracking-[0.3em] text-umbra-glow/70">The design tool</h2>
        <p className="mx-auto mt-3 max-w-2xl text-sm text-umbra-ink/60">
          A browser-based studio to design a shadow performance — sketch the silhouettes, cast them
          to the fleet, and solve the trajectories.
        </p>
      </div>

      <div className="space-y-12">
        {shots.map((s) => (
          <figure key={s.src}>
            <div className="overflow-hidden rounded-xl border border-white/10 bg-white/[0.02]">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={s.src} alt={s.alt} loading="lazy" decoding="async" className="w-full" />
            </div>
            <figcaption className="mt-3 text-center text-sm text-umbra-ink/60">
              <span className="font-medium text-umbra-ink/85">{s.title}.</span> {s.caption}
            </figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}
