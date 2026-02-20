import { Float32Vec, SpectralHistogram } from '../wasm/build/wasm_main_module';

// Size of the FFT window for spectrogram computation
export const SPECTROGRAM_SIZE = 2048;

// Simple spectrogram visualization
export class Spectrogram {
    canvas: HTMLCanvasElement;
    spectralHistogram: SpectralHistogram;
    histogramInputVec: Float32Vec;
    histogramOutputVec: Float32Vec;
    showFrequencies: boolean = false;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.spectralHistogram = new SpectralHistogram(SPECTROGRAM_SIZE);
        this.histogramInputVec = new Float32Vec(0);
        this.histogramOutputVec = new Float32Vec(0);
        this.resizeCanvas();
        new ResizeObserver(() => this.resizeCanvas).observe(canvas);
        canvas.addEventListener('click', () => {
            this.showFrequencies = !this.showFrequencies;
        });
    }

    // Return the amount of audio samples to get to render spectrogram.
    getSamples(): number {
        return SPECTROGRAM_SIZE;
    }

    // Draw a spectrogram visualization of audio data onto a canvas.
    draw(audioData: Float32Array, numChannels: number, sampleRate: number) {
        if (audioData.length !== SPECTROGRAM_SIZE * numChannels) {
            console.debug(`Passed ${audioData.length}, should be ${SPECTROGRAM_SIZE * numChannels}`);
        }
        this.histogramInputVec.set(audioData);
        this.spectralHistogram.compute(this.histogramInputVec, numChannels, this.histogramOutputVec);
        const spectrogram = this.histogramOutputVec.array;

        const canvasCtx = this.canvas.getContext('2d')!;

        // We use fixed norm instead of computing the max value and coloring the whole spectrogram depending on it.
        const NORM = 100.0;
        // const maxValue = spectrogram.reduce((v1, v2) => Math.max(v1, v2), 0.0);
        // console.debug(`Got ${maxValue}`);
        canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // We do not care about frequencies greater than 10Khz
        const spectrogramRange = (Math.min(1.0, 10000 / sampleRate) * SPECTROGRAM_SIZE) >>> 0;
        // console.debug(`Showing spectrogram up to ${spectrogramRange}`);
        const barWidth = this.canvas.width / spectrogramRange;
        canvasCtx.fillStyle = '#223344';
        for (let i = 0; i < spectrogramRange; i++) {
            const barHeight = Math.round((spectrogram[i] * this.canvas.height) / NORM);

            const x = i * barWidth;
            canvasCtx.fillRect(x, this.canvas.height - barHeight, barWidth, barHeight);
        }
        if (this.showFrequencies) {
            canvasCtx.font = 'bold 28px sans-serif';
            const metrics = canvasCtx.measureText('F2: 10000');
            const height = metrics.actualBoundingBoxAscent;
            const topFreq = findTop2(spectrogram);
            const binWidth = sampleRate / SPECTROGRAM_SIZE;
            const freq1 = ((topFreq[0] + 0.5) * binWidth) >>> 0;
            const freq2 = ((topFreq[1] + 0.5) * binWidth) >>> 0;
            canvasCtx.fillText(`F1: ${freq1}`, this.canvas.width - metrics.width * 2, 5 + height);
            canvasCtx.fillText(`F2: ${freq2}`, this.canvas.width - metrics.width * 1, 5 + height);
        }
    }

    // Clear the spectrogram canvas
    clear() {
        const canvasCtx = this.canvas.getContext('2d')!;
        canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    private resizeCanvas() {
        // See https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Optimizing_canvas#scaling_for_high_resolution_displays
        const { width, height } = this.canvas.getBoundingClientRect();
        this.canvas.width = width * window.devicePixelRatio;
        this.canvas.height = height * window.devicePixelRatio;
        this.canvas.style.width = `${width}px`;
        this.canvas.style.height = `${height}px`;
    }
}

// Return indices for top 2 values in array.
function findTop2(arr: Float32Array): number[] {
    let top1 = 0.0;
    let idx1 = 0;
    let top2 = 0.0;
    let idx2 = 0;
    for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        if (v > top1) {
            top2 = top1;
            idx2 = idx1;
            top1 = v;
            idx1 = i;
        } else if (v > top2) {
            top2 = v;
            idx2 = i;
        }
    }
    if (idx1 > idx2) {
        return [idx2, idx1];
    }
    return [idx1, idx2];
}
