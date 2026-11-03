import { describe, expect, it } from 'vitest';
import { fftSizeForSampleRate } from './utils';

describe('fftSizeForSampleRate', () => {
    it('test', () => {
        expect(fftSizeForSampleRate(44100, 40)).toBe(2048);
        expect(fftSizeForSampleRate(44100, 80)).toBe(4096);
        expect(fftSizeForSampleRate(48000, 40)).toBe(2048);
        expect(fftSizeForSampleRate(48000, 80)).toBe(4096);
        expect(fftSizeForSampleRate(96000, 40)).toBe(4096);
        expect(fftSizeForSampleRate(96000, 80)).toBe(8192);
        expect(fftSizeForSampleRate(48000, 1)).toBe(128);
    });
});
