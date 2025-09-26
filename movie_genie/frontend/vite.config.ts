import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // Build configuration for Flask integration
  build: {
    outDir: "../backend/dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, "index.html"),
      },
      output: {
        // Place JS and CSS files in Flask's static directory structure
        assetFileNames: (assetInfo) => {
          const extType = assetInfo.name?.split('.').pop();
          if (extType === 'css') {
            return 'css/[name]-[hash][extname]';
          }
          if (['png', 'jpg', 'jpeg', 'svg', 'gif', 'webp'].includes(extType || '')) {
            return 'img/[name]-[hash][extname]';
          }
          return 'assets/[name]-[hash][extname]';
        },
        chunkFileNames: 'js/[name]-[hash].js',
        entryFileNames: 'js/[name]-[hash].js',
      },
    },
  },
  // Configure base path for Flask static files
  base: mode === 'production' ? '/static/' : '/',
}));
