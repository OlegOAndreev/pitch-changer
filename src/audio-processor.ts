import * as Comlink from 'comlink';

import type { Float32RingBuffer } from './ring-buffer';

// AudioProcessor runs processing audio in the web worker. The output is written asynchronously into Float32RingBuffer.
// Closing the output ring buffer cancels the processing.

// The following are exported only for audio-modifier-worker.ts
export interface WorkerParams {
    processingMode: 'pitch' | 'time';
    pitchValue: number;
};

export interface WorkerApi {
    init(): Promise<boolean>;
    setParams(params: WorkerParams): void;
    // outputSab must be a buffer for Float32RingBuffer
    process(source: Float32Array, sampleRate: number, outputSab: SharedArrayBuffer): Promise<void>;
}

export class AudioProcessor {
    private worker: Worker;
    private proxy: Comlink.Remote<WorkerApi>;

    constructor() {
        this.worker = new Worker(new URL('./audio-processor-worker.ts', import.meta.url), {
            type: 'module',
        });
        this.proxy = Comlink.wrap(this.worker);
    }

    async init() {
        const result = await this.proxy.init();
        console.log(`Worker initialized with status ${result}`);
    }

    setParams(processingMode: 'pitch' | 'time', pitchValue: number) {
        this.proxy.setParams({
            processingMode: processingMode,
            pitchValue: pitchValue,
        });
    }

    async process(source: Float32Array, sampleRate: number, output: Float32RingBuffer): Promise<void> {
        await this.proxy.process(source, sampleRate, output.buffer);
    }
}
