/// Manager for the AudioProcessingWorker, see also AudioProcessorControl
import { AudioProcessorClient } from './audio-processor-client';
import type {
    AudioProcessorResponse,
    SetClientPortMessage,
    SetParamsMessage,
    WorkerInitMessage,
} from './audio-processor-types';
import audioProcessorURL from './audio-processor-worker.ts?worker&url';
import { logError } from './common-utils';
import type { ProcessingMode } from './types';
import { concatArrays } from './utils';

export class AudioProcessorManager {
    private worker: Worker | undefined;

    // Workaround for lack of async constructors
    static async create(): Promise<AudioProcessorManager> {
        const result = new AudioProcessorManager();
        await result.workerInit();
        return result;
    }

    private workerInit(): Promise<void> {
        this.worker = new Worker(audioProcessorURL, { type: 'module' });
        this.worker.onerror = (event: ErrorEvent) => {
            logError('AudioProcessorWorker', event);
        };

        let resolve: (value?: void) => void;
        const promise = new Promise<void>((res) => {
            resolve = res;
        });
        this.worker.onmessage = (event: MessageEvent<AudioProcessorResponse>) => {
            const message = event.data;
            // We can get only one message type from worker.
            if (message.type !== 'initDone') {
                return;
            }
            resolve();
        };

        const message: WorkerInitMessage = {
            type: 'init',
        };
        this.worker.postMessage(message);
        return promise;
    }

    // Creates a port which must be passed to AudioProcessorClient constructor
    createClientPort(): MessagePort {
        const channel = new MessageChannel();
        const message: SetClientPortMessage = {
            type: 'setClientPort',
            clientPort: channel.port1,
        };
        this.worker!.postMessage(message, [channel.port1]);
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
        this.worker!.postMessage(message);
    }

    // Terminate the worker.
    terminate(): void {
        this.worker!.terminate();
    }

    // A helper which processes the data all at once.
    async processAudio(data: Float32Array): Promise<Float32Array> {
        const chunks: Float32Array[] = [];
        let resolve: (value: Float32Array) => void;
        const promise = new Promise<Float32Array>((res) => {
            resolve = res;
        });

        const port = this.createClientPort();
        const client = new AudioProcessorClient(port, (samples, finished) => {
            console.log(`Got ${samples.length} samples`);
            chunks.push(samples);
            if (finished) {
                const result = concatArrays(chunks);
                resolve(result);
            }
        });
        client.processSamples(data.slice());
        client.finish();
        return promise;
    }
}
