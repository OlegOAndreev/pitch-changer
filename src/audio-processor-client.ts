// AudioProcessorClient allows controlling AudioProcessorWorker from another Worker/AudioWorklet. It wraps a
// MessagePort and provides request/response messaging for sample processing: source sample buffers are transferred from
// the client to the worker, processed sample buffers are transferred from the worker to the client.
import type {
    FinishProcessRequest,
    ProcessSamplesRequest,
    ProcessSamplesResponse,
    ResetRequest,
} from './audio-processor-worker';

export class AudioProcessorClient {
    private port: MessagePort;
    private onProcessedSamples: (samples: Float32Array, finished: boolean) => void;

    // Construct from the message port (get the port via AudioProcessorManager.getControlClientPort) and the callback.
    // onProcessedSamples is called when the workers finishes processing samples (either after processSamples or finish
    // is called).
    constructor(port: MessagePort, onProcessedSamples: (samples: Float32Array, finished: boolean) => void) {
        this.port = port;
        this.onProcessedSamples = onProcessedSamples;
        this.port.onmessage = (event: MessageEvent<ProcessSamplesResponse>) => {
            const message = event.data;
            if (message.type !== 'processSamplesResponse') {
                return;
            }
            this.onProcessedSamples(message.samples, message.finished);
        };
    }

    // Resets the processor, should be called before processing new audio
    reset(): void {
        const message: ResetRequest = {
            type: 'resetRequest',
        };
        this.port.postMessage(message);
    }

    // Request processing samples from the worker. The `samples` buffer will be transferred to avoid copying.
    processSamples(samples: Float32Array): void {
        const message: ProcessSamplesRequest = {
            type: 'processSamplesRequest',
            samples,
        };
        this.port.postMessage(message, [samples]);
    }

    // Complete processing, previously sent samples will be flushed.
    finish(): void {
        const message: FinishProcessRequest = {
            type: 'finishRequest',
        };
        this.port.postMessage(message);
    }
}
