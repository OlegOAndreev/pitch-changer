import initWasmModule, { Float32Vec, SpectralHistogram } from "../wasm/build/wasm_main_module";

// Size of the FFT window for spectrogram computation
export const SPECTROGRAM_SIZE = 2048;

await initWasmModule();

const spectralHistogram = new SpectralHistogram(SPECTROGRAM_SIZE);
const histogramInputVec = new Float32Vec(0);
const histogramOutputVec = new Float32Vec(0);

// Draw a spectrogram visualization of audio data onto a canvas
export function drawSpectrogram(canvas: HTMLCanvasElement, audioData: Float32Array, numChannels: number) {
    histogramInputVec.set(audioData);
    spectralHistogram.compute(histogramInputVec, numChannels, histogramOutputVec);
    const spectrogram = histogramOutputVec.array;

    const { width, height } = canvas.getBoundingClientRect();
    const canvasCtx = canvas.getContext('2d')!;

    // We use fixed norm instead of computing the max value and coloring the whole spectrogram depending on it.
    const NORM = 100.0;
    // const maxValue = spectrogram.reduce((v1, v2) => Math.max(v1, v2), 0.0);
    // console.debug(`Got ${maxValue}`);
    canvasCtx.clearRect(0, 0, width, height);

    // We do not care about frequencies greater than ~11Khz
    const spectrogramRange = spectrogram.length / 2;
    const barWidth = width / spectrogramRange;
    canvasCtx.fillStyle = '#444444';
    for (let i = 0; i < spectrogramRange; i++) {
        const barHeight = Math.round(spectrogram[i] * height / NORM);

        const x = i * barWidth;
        canvasCtx.fillRect(x, height - barHeight, barWidth, barHeight);
    }
}

// Clear the spectrogram canvas
export function clearSpectrogram(canvas: HTMLCanvasElement) {
    const { width, height } = canvas.getBoundingClientRect();
    const canvasCtx = canvas.getContext('2d')!;
    canvasCtx.clearRect(0, 0, width + 1.0, height + 1.0);
}
