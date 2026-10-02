import type { UserConfig } from 'vite';

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
