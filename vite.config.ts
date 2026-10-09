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
} satisfies UserConfig;
