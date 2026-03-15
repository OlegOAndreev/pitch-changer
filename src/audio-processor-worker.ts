// This is a worker for AudioProcessor, see also AudioProcessorControl
import initWasmModule, {
    Float32Vec,
    get_settings,
    MultiPitchShifter,
    PitchShiftParams,
} from '../wasm/build/wasm_main_module';
import type { ProcessingMode } from './types';

// Message types for communication with the AudioProcessor
export interface WorkerParams {
    processingMode: ProcessingMode;
    pitchValue: number;
    sampleRate: number;
    numChannels: number;
    fftSize: number;
}

export interface SetClientPortMessage {
    type: 'setClientPort';
    clientPort: MessagePort;
}

export interface SetParamsMessage {
    type: 'setParams';
    params: WorkerParams;
}

export interface ResetRequest {
    type: 'resetRequest';
}

export interface ProcessSamplesRequest {
    type: 'processSamplesRequest';
    samples: Float32Array;
}

export interface FinishProcessRequest {
    type: 'finishRequest';
}

// This message is sent in response both to ProcessSamplesRequest and FinishProcessRequest
export interface ProcessSamplesResponse {
    type: 'processSamplesResponse';
    samples: Float32Array;
    finished: boolean;
}

type AudioProcessorRequest = SetClientPortMessage | SetParamsMessage;
type AudioProcessorClientRequest = ResetRequest | ProcessSamplesRequest | FinishProcessRequest;

class AudioProcessorWorker {
    private params: WorkerParams = {
        processingMode: 'pitch',
        pitchValue: 1.0,
        sampleRate: 44100,
        numChannels: 1,
        fftSize: 4096,
    };
    private paramsDirty: boolean = false;
    private processor: MultiPitchShifter | null = null;
    private processorNumChannels: number | null = null;
    private inputVec: Float32Vec = new Float32Vec(0);
    private processedVec: Float32Vec = new Float32Vec(0);
    private clientPort: MessagePort | undefined;

    // Initialize WASM module
    async init(): Promise<void> {
        const module = await initWasmModule();
        const wasmMemory = module.memory;
        console.debug(`Player worker: Wasm settings: ${get_settings()}`);
        console.debug(`Player worker: Initial wasm memory size: ${wasmMemory.buffer.byteLength}`);
    }

    setClientPort(clientPort: MessagePort) {
        if (this.clientPort) {
            this.clientPort.close();
        }
        this.clientPort = clientPort;
        // Start the client event loop
        this.clientPort.onmessage = (event: MessageEvent<AudioProcessorClientRequest>) => {
            this.onMessage(event.data);
        };
    }

    setParams(newParams: WorkerParams): void {
        this.params = newParams;
        this.paramsDirty = true;
    }

    private onMessage(message: AudioProcessorClientRequest) {
        switch (message.type) {
            case 'resetRequest':
                this.reset();
                break;

            case 'processSamplesRequest':
                this.processSamples(message.samples);
                break;

            case 'finishRequest':
                this.finish();
                break;

            default:
                console.error('Player worker: Unknown message type:', message);
                break;
        }
    }

    private reset() {
        const processor = this.getProcessor();
        processor.reset();
    }

    private processSamples(samples: Float32Array) {
        const processor = this.getProcessor();
        this.inputVec.set(samples);
        this.processedVec.clear();
        processor.process(this.inputVec, this.processedVec);
        // Clone the array: we will transfer the buffer back and allow processedVec to be reused.
        const result = this.processedVec.array.slice();
        // console.debug(
        //     `AudioProcessorWorker: Processed ${samples.length} input samples into ${result.length} output samples`,
        // );
        this.sendResult(result, false);
    }

    private finish() {
        const processor = this.getProcessor();
        this.processedVec.clear();
        processor.finish(this.processedVec);
        const result = this.processedVec.array.slice();
        // console.debug(`AudioProcessorWorker: Finished into ${result.length} output samples`);
        this.sendResult(result, true);
    }

    private getProcessor(): MultiPitchShifter {
        if (this.processor && !this.paramsDirty) {
            return this.processor;
        }

        let timeStretch;
        let pitchShift;
        switch (this.params.processingMode) {
            case 'formant-preserving-pitch':
            case 'pitch':
                timeStretch = 1.0;
                pitchShift = this.params.pitchValue;
                break;
            case 'time':
                timeStretch = this.params.pitchValue;
                pitchShift = 1.0;
                break;
            default:
                throw new Error(`Unknown processing mode ${this.params.processingMode}`);
        }

        const params = new PitchShiftParams(this.params.sampleRate, pitchShift, timeStretch);
        params.fft_size = this.params.fftSize;
        try {
            if (this.params.processingMode === 'formant-preserving-pitch') {
                params.quefrency_cutoff = 1.0;
            }

            if (this.processor && this.processorNumChannels === this.params.numChannels) {
                console.log(`Player worker: Updating processor parameters to ${params.to_debug_string()}`);
                this.processor.update_params(params);
            } else {
                console.log(`Player worker: Creating new processor with parameters ${params.to_debug_string()}`);
                if (this.processor) {
                    this.processor.free();
                }
                this.processor = new MultiPitchShifter(params, this.params.numChannels);
            }
        } finally {
            params.free();
        }

        this.processorNumChannels = this.params.numChannels;
        this.paramsDirty = false;

        return this.processor!;
    }

    private sendResult(samples: Float32Array, finished: boolean) {
        if (!this.clientPort) {
            throw new Error('Impossible: init should be called before before all other');
        }

        const response: ProcessSamplesResponse = {
            type: 'processSamplesResponse',
            samples,
            finished,
        };
        this.clientPort.postMessage(response, [samples.buffer]);
    }
}

async function init() {
    const workerImpl = new AudioProcessorWorker();
    await workerImpl.init();
    // Start manager event loop
    onmessage = (event: MessageEvent<AudioProcessorRequest>) => {
        const message = event.data;
        switch (message.type) {
            case 'setClientPort':
                workerImpl.setClientPort(message.clientPort);
                break;

            case 'setParams':
                workerImpl.setParams(message.params);
                break;

            default:
                console.error('Player worker: Unknown message type:', message);
                break;
        }
    };
}

init();
