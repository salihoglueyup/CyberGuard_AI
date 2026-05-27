import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    // react-globe.gl + three-globe currently produce a large but isolated vendor chunk.
    chunkSizeWarningLimit: 1300,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return;

          if (id.includes('react-globe.gl')) return 'vendor-globe';
          if (id.includes('three-globe')) return 'vendor-three-globe';
          if (id.includes('@react-three/fiber') || id.includes('@react-three/drei')) return 'vendor-r3f';
          if (id.includes('/three/')) return 'vendor-three';

          if (id.includes('/react-dom/') || id.includes('/react-router-dom/') || id.includes('/react/')) return 'vendor-react';
          if (id.includes('/recharts/')) return 'vendor-charts';
          if (id.includes('/framer-motion/')) return 'vendor-motion';
          if (id.includes('/react-grid-layout/')) return 'vendor-grid';
          if (id.includes('/leaflet/') || id.includes('/react-leaflet/')) return 'vendor-map';
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.js',
    css: true,
    include: ['src/**/*.{test,spec}.{js,jsx}'],
  }
})
