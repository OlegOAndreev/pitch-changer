import * as Comlink from 'comlink';

import type { Float32RingBuffer } from './sync';
import type { InterleavedAudio } from './types';

// The following are exported only for audio-modifier-worker.ts
export interface WorkerParams {
    processingMode: 'pitch' | 'time';
    pitchValue: number;
};

export interface WorkerApi {
    init(): Promise<boolean>;
    setParams(params: WorkerParams): void;
    // outputSab must be a buffer for Float32RingBuffer
    process(source: InterleavedAudio, outputSab: SharedArrayBuffer): Promise<void>;
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
    // data is processed, or output is closed.
    async process(source: InterleavedAudio, output: Float32RingBuffer): Promise<void> {
        await this.proxy.process(source, output.buffer);
    }
}
