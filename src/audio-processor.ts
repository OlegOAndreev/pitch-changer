/// Manager for the AudioProcessingWorker, see also AudioProcessorControl
import type { SetClientPortMessage, SetParamsMessage } from './audio-processor-worker';
import type { ProcessingMode } from './types';

export class AudioProcessorManager {
    private worker: Worker;

    constructor() {
        this.worker = new Worker(new URL('./audio-processor-worker.ts', import.meta.url), { type: 'module' });
        this.worker.onerror = (error) => {
            console.error('AudioProcessorManager: Worker error:', error);
        };
    }

    // Creates a port which must be passed to AudioProcessorClient constructor
    createClientPort(): MessagePort {
        const channel = new MessageChannel();
        const message: SetClientPortMessage = {
            type: 'setClientPort',
            clientPort: channel.port1,
        };
        this.worker.postMessage(message, [channel.port1]);
        return channel.port2;
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
                processingMode,
                pitchValue,
                sampleRate,
                numChannels,
                fftSize,
            },
        };
        this.worker.postMessage(message);
    }

    // Terminate the worker.
    terminate(): void {
        this.worker.terminate();
    }
}
