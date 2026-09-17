import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Relative base so the built site works from any path (static host, GitHub Pages, file server).
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        // Split the rarely-changing libraries so a content edit does not
        // invalidate the whole bundle in the browser cache.
        manualChunks: {
          react: ['react', 'react-dom'],
          katex: ['katex'],
          d3: ['d3-scale', 'd3-shape'],
        },
      },
    },
  },
});
