import type { ProcessingMode } from './types';
import type {
    ProcessedSamplesMessage,
    ProcessSamplesMessage,
    SetParamsMessage,
    FinishMessage,
} from './audio-processor-worker';

// AudioProcessorManager start a worker which processes audio on demand.
export class AudioProcessorManager {
    private worker: Worker;
    private processedCallback: ((samples: Float32Array, finished: boolean) => void) | null = null;

    constructor() {
        this.worker = new Worker(new URL('./audio-processor-worker.ts', import.meta.url), { type: 'module' });

        this.worker.onmessage = (event: MessageEvent<ProcessedSamplesMessage>) => {
            const message = event.data;
            if (message.type != 'processedSamples') {
                console.error(`Unknown message type ${message.type}`);
                return;
            }
            if (this.processedCallback) {
                this.processedCallback(message.samples, message.finished);
            }
        };

        // Handle worker errors
        this.worker.onerror = (error) => {
            console.error('AudioProcessorManager: Worker error:', error);
        };
    }

    // Set callback that is called each time the new samples are done processing (they are processed sequentially in
    // response to processSamples). If finished is true, this is the last samples flushed after finish() is called.
    setProcessedCallback(callback: (samples: Float32Array, finished: boolean) => void) {
        this.processedCallback = callback;
    }

    // Update processing params. This method may be called at any time.
    setParams(
        processingMode: ProcessingMode,
        pitchValue: number,
        sampleRate: number,
        numChannels: number,
        fftSize: number,
    ): void {
        const message: SetParamsMessage = {
            type: 'setParams',
            params: {
                processingMode: processingMode,
                pitchValue: pitchValue,
                sampleRate: sampleRate,
                numChannels: numChannels,
                fftSize: fftSize,
            },
        };
        this.worker.postMessage(message);
    }

    // Request processing samples from the worker. NOTE: The samples buffer will be transferred to the worker.
    processSamples(samples: Float32Array) {
        const message: ProcessSamplesMessage = {
            type: 'processSamples',
            samples: samples,
        };
        this.worker.postMessage(message, [samples.buffer]);
    }

    // Complete processing. The previously sent samples will be flushed.
    finish() {
        const message: FinishMessage = {
            type: 'finish',
        };
        this.worker.postMessage(message);
    }

    // Terminate the worker
    terminate(): void {
        this.worker.terminate();
    }
}
