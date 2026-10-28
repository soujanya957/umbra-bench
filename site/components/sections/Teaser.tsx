export default function Teaser() {
  return (
    <section id="video" className="mx-auto max-w-4xl px-6 py-24">
      <h2 className="mb-8 text-center text-xs font-semibold uppercase tracking-[0.3em] text-umbra-glow/70">
        Video
      </h2>

      {/* 16:9 placeholder. Swap in a <video> or <iframe> (YouTube/Vimeo) when ready. */}
      <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-umbra-glow/15 bg-black/40">
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-full border border-umbra-glow/30">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="currentColor" className="ml-1 text-umbra-glow/70">
              <path d="M8 5v14l11-7z" />
            </svg>
          </div>
          <p className="text-sm tracking-[0.2em] text-umbra-glow/50 uppercase">Teaser coming soon</p>
        </div>
      </div>

      <p className="mt-4 text-center text-xs text-umbra-ink/40">
        {/* Example: <iframe> a YouTube embed, or a self-hosted <video src="/teaser.mp4" controls /> */}
        Replace this block with the project teaser video.
      </p>
    </section>
  );
}
