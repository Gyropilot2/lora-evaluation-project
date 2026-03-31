import { readFileSync } from "node:fs";

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const runtimeConfig = JSON.parse(
  readFileSync(new URL("../runtime_config.json", import.meta.url), "utf8"),
) as {
  backend: { host: string; port: number };
  frontend: { host: string; port: number };
};

const backendOrigin = `http://${runtimeConfig.backend.host}:${runtimeConfig.backend.port}`;

export default defineConfig({
  plugins: [react()],
  server: {
    host: runtimeConfig.frontend.host,
    port: runtimeConfig.frontend.port,
    proxy: {
      "/api": backendOrigin,
    },
  },
  preview: {
    host: runtimeConfig.frontend.host,
    port: runtimeConfig.frontend.port,
  },
});
