import type { UserConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default {
    // For github pages
    base: '/pitch-changer/',

    worker: {
        format: 'es',
    },

    build: {
        chunkSizeWarningLimit: 1500,
    },

    server: {
        headers: {
            'Cross-Origin-Embedder-Policy': 'require-corp',
            'Cross-Origin-Opener-Policy': 'same-origin',
        },
    },
} satisfies UserConfig;
