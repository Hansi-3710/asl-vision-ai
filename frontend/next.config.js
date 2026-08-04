/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // "standalone" produces a self-contained .next/standalone/server.js
  // bundle with only the production dependencies it actually needs --
  // required by Dockerfile's final runtime stage (which only copies
  // .next/standalone + .next/static, not the full node_modules).
  output: "standalone",
};

module.exports = nextConfig;
