import { execSync } from 'node:child_process';
import type { UserConfig } from 'vite';

function getBuildCommit(): string {
    const commit =
        process.env.GIT_COMMIT ?? execSync('git rev-parse HEAD 2>/dev/null || echo unknown', { encoding: 'utf8' });
    return commit.trim().slice(0, 8);
}

const buildDate = new Date().toISOString();

export default {
    // For github pages
    base: '/pitch-changer/',

    define: {
        'import.meta.env.VITE_BUILD_COMMIT': JSON.stringify(getBuildCommit()),
        'import.meta.env.VITE_BUILD_DATE': JSON.stringify(buildDate),
    },

    worker: {
        format: 'es',
    },

    build: {
        chunkSizeWarningLimit: 1500,
    },
} satisfies UserConfig;
