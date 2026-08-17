import type { NextConfig } from "next";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      // Rollout videos and batch thumbnails are served by FastAPI, because
      // Antioch's own artifact URLs are signed and expire. Proxying them
      // keeps every URL in the event stream a plain relative path, so the
      // same event works whether it came from the fixture or the backend.
      { source: "/artifacts/:path*", destination: `${API}/artifacts/:path*` },
    ];
  },
};

export default nextConfig;
