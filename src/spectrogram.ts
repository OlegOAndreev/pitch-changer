import { Float32Vec, SpectralHistogram } from "../wasm/build/wasm_main_module";

// Size of the FFT window for spectrogram computation
export const SPECTROGRAM_SIZE = 2048;

let spectralHistogram: SpectralHistogram | null;
let histogramInputVec: Float32Vec | null;
let histogramOutputVec: Float32Vec | null;


function resizeCanvas(canvas: HTMLCanvasElement) {
    // See https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Optimizing_canvas#scaling_for_high_resolution_displays
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
}

// Sets up event for canvas.
export function setupCanvasEvents(canvas: HTMLCanvasElement) {
    resizeCanvas(canvas);
    canvas.addEventListener('resize', () => resizeCanvas(canvas));
}

// Draw a spectrogram visualization of audio data onto a canvas
export function drawSpectrogram(canvas: HTMLCanvasElement, audioData: Float32Array, numChannels: number) {
    if (!spectralHistogram || !histogramInputVec || !histogramOutputVec) {
        spectralHistogram = new SpectralHistogram(SPECTROGRAM_SIZE);
        histogramInputVec = new Float32Vec(0);
        histogramOutputVec = new Float32Vec(0);
    }
    histogramInputVec.set(audioData);
    spectralHistogram.compute(histogramInputVec, numChannels, histogramOutputVec);
    const spectrogram = histogramOutputVec.array;

    const canvasCtx = canvas.getContext('2d')!;

    // We use fixed norm instead of computing the max value and coloring the whole spectrogram depending on it.
    const NORM = 100.0;
    // const maxValue = spectrogram.reduce((v1, v2) => Math.max(v1, v2), 0.0);
    // console.debug(`Got ${maxValue}`);
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    // We do not care about frequencies greater than ~11Khz
    const spectrogramRange = spectrogram.length / 2;
    const barWidth = canvas.width / spectrogramRange;
    canvasCtx.fillStyle = '#223344';
    for (let i = 0; i < spectrogramRange; i++) {
        const barHeight = Math.round(spectrogram[i] * canvas.height / NORM);

        const x = i * barWidth;
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
    }
}

// Clear the spectrogram canvas
export function clearSpectrogram(canvas: HTMLCanvasElement) {
    const canvasCtx = canvas.getContext('2d')!;
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
}
