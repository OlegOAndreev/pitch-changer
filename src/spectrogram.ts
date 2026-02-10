export function drawSpectrogram(canvas: HTMLCanvasElement, spectrogram: Float32Array) {
    const { width, height } = canvas.getBoundingClientRect();
    const canvasCtx = canvas.getContext('2d')!;

    const maxValue = spectrogram.reduce((v1, v2) => Math.max(v1, v2), 0.0);
    const norm = Math.min(maxValue, 100.0);
    // console.debug(`Got ${maxValue}`);
    canvasCtx.clearRect(0, 0, width, height);

    // We do not care about frequencies greater than ~11Khz
    const spectrogramRange = spectrogram.length / 2;
    const barWidth = width / spectrogramRange;
    canvasCtx.fillStyle = '#444444';
    for (let i = 0; i < spectrogramRange; i++) {
        const barHeight = spectrogram[i] * height / norm;

        const x = i * barWidth;
        canvasCtx.fillRect(x, height - barHeight, barWidth, barHeight);
    }
}

export function clearSpectrogram(canvas: HTMLCanvasElement) {
    const { width, height } = canvas.getBoundingClientRect();
    const canvasCtx = canvas.getContext('2d')!;
    canvasCtx.clearRect(0, 0, width, height + 0.5);
}
