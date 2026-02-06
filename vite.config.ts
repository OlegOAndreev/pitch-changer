import type { UserConfig } from 'vite'
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default {
  // For github pages
  base: '/pitch-changer/',

  worker: {
    format: 'es',
  },

  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/coi-serviceworker/coi-serviceworker.min.js",
          dest: ".",
        },
      ],
    })
  ],

  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin'
    }
  },
} satisfies UserConfig
