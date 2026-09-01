// AudioProcessorClient allows controlling AudioProcessorWorker from another Worker/AudioWorklet. It wraps a
// MessagePort and provides request/response messaging for sample processing: source sample buffers are transferred from
// the client to the worker, processed sample buffers are transferred from the worker to the client.
import type {
    FinishProcessRequest,
    ProcessSamplesRequest,
    ProcessSamplesResponse,
    ResetRequest,
    SetParamsMessage,
} from './audio-processor-types';
import type { ProcessingMode } from './types';

export class AudioProcessorClient {
    private port: MessagePort;

    // Construct from the message port (get the port via AudioProcessorManager.getControlClientPort) and the callback.
    // onProcessedSamples is called when the workers finishes processing samples (either after processSamples or finish
    // is called).
    constructor(port: MessagePort, onProcessedSamples: (samples: Float32Array, finished: boolean) => void) {
        this.port = port;
        this.port.onmessage = (event: MessageEvent<ProcessSamplesResponse>) => {
            const message = event.data;
            if (message.type !== 'processSamplesResponse') {
                return;
            }
            onProcessedSamples(message.samples, message.finished);
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
        this.port.postMessage(message, [samples.buffer]);
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
        this.port.postMessage(message);
    }

    // Complete processing, previously sent samples will be flushed.
    finish(): void {
        const message: FinishProcessRequest = {
            type: 'finishRequest',
        };
        this.port.postMessage(message);
    }
}
