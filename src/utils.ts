// Compute the minimal fft size for given sample rate which covers window size (in milliseconds).
export function fftSizeForSampleRate(sampleRate: number, windowSizeMs: number): number {
    let result = 128;
    const targetSize = (sampleRate * windowSizeMs) / 1000;
    while (result < targetSize) {
        result *= 2;
    }
    return result;
}
