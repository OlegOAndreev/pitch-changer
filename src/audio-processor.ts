import * as Comlink from 'comlink';

import type { Float32RingBuffer } from './sync';
import type { InterleavedAudio } from './types';

// The following are exported only for audio-modifier-worker.ts
export interface WorkerParams {
    processingMode: 'pitch' | 'time';
    pitchValue: number;
};

export type HistogramCallback = ((histogram: Float32Array) => void) | null;

export interface WorkerApi {
    init(): Promise<boolean>;
    setParams(params: WorkerParams): void;
    // outputSab must be a buffer for Float32RingBuffer
    process(input: InterleavedAudio, outputSab: SharedArrayBuffer, histogramCallback: HistogramCallback): Promise<void>;
}

// AudioProcessor runs processing audio in the web worker. The output is written asynchronously into Float32RingBuffer.
// Closing the output ring buffer cancels the processing.
export class AudioProcessor {
    private worker: Worker;
    private proxy: Comlink.Remote<WorkerApi>;

    constructor() {
        this.worker = new Worker(new URL('./audio-processor-worker.ts', import.meta.url), {
            type: 'module',
        });
        this.proxy = Comlink.wrap(this.worker);
    }

    // init must be called before any other parameters.
    async init() {
        const result = await this.proxy.init();
        console.log(`Worker initialized with status ${result}`);
    }

    // Update processing params. This method may be called even if process() is still running.
    setParams(processingMode: 'pitch' | 'time', pitchValue: number) {
        this.proxy.setParams({
            processingMode: processingMode,
            pitchValue: pitchValue,
        });
    }

    // Run processing source audio data and store the result into output. The processing stops when either all source
    // data is processed, or output is closed. If histogramCallback is not null, it will periodically be called with
    // current output spectrogram.
    async process(input: InterleavedAudio, output: Float32RingBuffer, histogramCallback: HistogramCallback): Promise<void> {
        let proxyCallback = null;
        if (histogramCallback != null) {
            proxyCallback = Comlink.proxy(histogramCallback);
        }
        await this.proxy.process(input, output.buffer, proxyCallback);
    }
}
