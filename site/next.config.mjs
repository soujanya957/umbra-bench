/** @type {import('next').NextConfig} */

// Static export so the site deploys anywhere (GitHub Pages, Netlify, S3, a plain folder).
// For GitHub *project* pages served at https://<user>.github.io/<repo>/, set the base path:
//   BASE_PATH=/umbra-site npm run build
// For a user/custom-domain deploy, leave BASE_PATH unset.
const basePath = process.env.BASE_PATH ?? "";

const nextConfig = {
  output: "export",
  basePath,
  assetPrefix: basePath || undefined,
  images: { unoptimized: true }, // required for `output: export`
  trailingSlash: true,
  // Exposed to the browser so raw <img src> to /public assets can be prefixed
  // with the base path on GitHub Pages (Next doesn't auto-prefix string src).
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default nextConfig;
