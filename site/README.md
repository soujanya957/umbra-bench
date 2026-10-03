# UMBRA — project website

Standalone static site for the UMBRA project (multi-robot shadow art). It is
**separate from the robot console** (`../`, the Next app that drives the arms):
this one is public, static, and deploys anywhere; the console needs a local
backend + hardware and can't be cloud-hosted.

Landing animation: a hanging pull cord — pull it to turn on the light, revealing
the UMBRA title, a lit robot arm, and its cast shadow. Below the fold: Abstract
and Video sections (placeholders to fill in).

## Develop

```bash
cd umbra-site
npm install
npm run dev        # http://localhost:4000
```

## Build (static export)

```bash
npm run build      # → ./out  (fully static)
```

Serve `out/` with any static host, or preview locally:

```bash
npx serve out
```

## Deploy to GitHub Pages

- **User / custom-domain site** (served at the domain root): just publish `out/`.
- **Project page** (served at `https://<user>.github.io/<repo>/`): set the base
  path so assets resolve:

  ```bash
  BASE_PATH=/<repo> npm run build
  ```

  Then push `out/` to the `gh-pages` branch (or point a GitHub Actions Pages
  workflow at it). Add an empty `.nojekyll` file in `out/` so Next's `_next/`
  folder isn't stripped by Jekyll.

## Where to edit

| Thing | File |
|------|------|
| Hero / light-cord animation | `components/LightCordHero.tsx` |
| Robot + shadow illustration | `RobotAndShadow` in `components/LightCordHero.tsx` |
| Abstract text | `components/sections/Abstract.tsx` |
| Video embed | `components/sections/Teaser.tsx` |
| Colors / theme | `tailwind.config.ts` |
| Page composition | `app/page.tsx` |
