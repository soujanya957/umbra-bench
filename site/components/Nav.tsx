const links = [
  { href: "#abstract", label: "Project overview" },
  { href: "#video", label: "Demos" },
  { href: "#design", label: "The design tool" },
];

export default function Nav() {
  return (
    <nav className="fixed left-0 top-0 z-40 flex flex-wrap gap-2 p-4">
      {links.map((l) => (
        <a
          key={l.href}
          href={l.href}
          className="rounded-full border border-white/15 bg-black/20 px-3 py-1.5 text-[11px] uppercase tracking-[0.12em] text-umbra-ink/75 backdrop-blur-sm transition-colors hover:border-white/35 hover:bg-white/10 hover:text-white"
        >
          {l.label}
        </a>
      ))}
    </nav>
  );
}
