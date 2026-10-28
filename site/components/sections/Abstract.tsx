export default function Abstract() {
  return (
    <section id="abstract" className="mx-auto max-w-3xl px-6 py-24">
      <h2 className="mb-6 text-center text-xs font-semibold uppercase tracking-[0.3em] text-umbra-glow/70">
        Project overview
      </h2>
      <div className="space-y-5 text-lg leading-relaxed text-umbra-ink/85">
        <p>
          UMBRA is a system for designing and performing multi-robot shadow art. A fleet of small
          robot arms each hold a shape or object, and under a single light source their overlapping
          silhouettes are cast onto a screen as one designed shadow — a letter, a figure, or an
          animated scene.
        </p>
        <p>
          Given a target image or animation, UMBRA assigns shapes to the arms and solves for the
          joint trajectories that make their combined shadow reproduce the intended picture, then
          coordinates the fleet to perform it over time. The toolchain covers the full loop:
          calibrating each arm against its simulation so the real pose matches the model, authoring
          target silhouettes and keyframes, optimizing trajectories against the desired shadow, and
          driving the physical arms to stage the performance.
        </p>
      </div>
    </section>
  );
}
