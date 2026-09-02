import { defineConfig } from "vite";
import preact from "@preact/preset-vite";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  base: "/ui/",
  plugins: [preact()],
  build: {
    outDir: fileURLToPath(new URL("../pyagent/webui/dist", import.meta.url)),
    emptyOutDir: true,
    assetsDir: "assets",
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/version": "http://127.0.0.1:8000",
      "/profiles": "http://127.0.0.1:8000",
      "/skills": "http://127.0.0.1:8000",
      "/tools": "http://127.0.0.1:8000",
      "/extensions": "http://127.0.0.1:8000",
      "/prompts": "http://127.0.0.1:8000",
      "/run": "http://127.0.0.1:8000",
      "/agents": "http://127.0.0.1:8000",
    },
  },
  test: {
    environment: "node",
  },
});
