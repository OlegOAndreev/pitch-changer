// This is a worker for AudioProcessor, see also AudioProcessorControl
import initWasmModule, {
    Float32Vec,
    get_settings,
    MultiPitchShifter,
    PitchShiftParams,
} from '../wasm/build/wasm_main_module';
import type {
    AudioProcessorClientRequest,
    AudioProcessorRequest,
    ProcessSamplesResponse,
    WorkerInitResponse,
    WorkerParams,
} from './audio-processor-types';

class AudioProcessorWorker {
    private params: WorkerParams = {
        processingMode: 'pitch',
        pitchValue: 1.0,
        sampleRate: 44100,
        numChannels: 1,
        fftSize: 4096,
    };
    private workerIsInit = false;
    private paramsAreDirty = false;
    private wasmInit: Promise<void> | undefined;
    private processor: MultiPitchShifter | undefined;
    private processorNumChannels: number | undefined;
    private inputVec: Float32Vec | undefined;
    private processedVec: Float32Vec | undefined;
    private clientPort: MessagePort | undefined;

    // Worker requires a separate message for async initialization, we can't use top-level await for initialization,
    // otherwise the messages get dropped:
    // https://stackoverflow.com/questions/34409254/are-messages-sent-via-worker-postmessage-queued
    // https://github.com/GoogleChromeLabs/comlink/issues/635
    init() {
        this.workerIsInit = true;

        const message: WorkerInitResponse = {
            type: 'initDone',
        };
        postMessage(message);
    }

    setClientPort(clientPort: MessagePort) {
        if (!this.workerIsInit) {
            throw new Error('Worker not initialized before calling methods');
        }

        if (this.clientPort) {
            this.clientPort.close();
        }
        this.clientPort = clientPort;
        // Start the client event loop
        console.log(`Starting client loop`);
        this.clientPort.onmessage = (event: MessageEvent<AudioProcessorClientRequest>) => {
            this.onMessage(event.data);
        };
    }

    setParams(newParams: WorkerParams): void {
        if (!this.workerIsInit) {
            throw new Error('Worker not initialized before calling methods');
        }

        this.params = newParams;
        this.paramsAreDirty = true;
    }

    private async onMessage(message: AudioProcessorClientRequest) {
        switch (message.type) {
            case 'resetRequest':
                await this.reset();
                break;

            case 'processSamplesRequest':
                await this.processSamples(message.samples);
                break;

            case 'finishRequest':
                await this.finish();
                break;

            default:
                console.error('Player worker: Unknown message type:', message);
                break;
        }
    }

    // Lazily initialize WASM module: this is important for extension where we want to postpone initialization as far as
    // possible. Do not run initWasmModule more than once.
    private async ensureWasmInit(): Promise<void> {
        if (!this.wasmInit) {
            this.wasmInit = this.initWasm();
        }
        return this.wasmInit;
    }

    private async initWasm() {
        const module = await initWasmModule();
        const wasmMemory = module.memory;
        console.debug(`AudioProcessorWorker: Wasm settings: ${get_settings()}, initial wasm memory size ${wasmMemory.buffer.byteLength}`);
    }

    private async reset() {
        if (!this.workerIsInit) {
            throw new Error('Worker not initialized before calling methods');
        }

        await this.ensureWasmInit();
        const processor = this.getProcessor();
        processor.reset();
    }

    private async processSamples(samples: Float32Array) {
        if (!this.workerIsInit) {
            throw new Error('Worker not initialized before calling methods');
        }

        await this.ensureWasmInit();
        if (!this.inputVec) {
            this.inputVec = new Float32Vec(0);
        }
        this.inputVec.set(samples);
        if (!this.processedVec) {
            this.processedVec = new Float32Vec(0);
        }
        this.processedVec.clear();

        const processor = this.getProcessor();
        processor.process(this.inputVec, this.processedVec);
        // Clone the array: we will transfer the buffer back to the client and allow processedVec to be reused.
        const result = this.processedVec.array.slice();
        // console.debug(
        //     `AudioProcessorWorker: Processed ${samples.length} input samples into ${result.length} output samples`,
        // );
        this.sendResult(result, false);
    }

    private async finish() {
        if (!this.workerIsInit) {
            throw new Error('Worker not initialized before calling methods');
        }

        await this.ensureWasmInit();
        if (!this.processedVec) {
            this.processedVec = new Float32Vec(0);
        }
        this.processedVec.clear();

        const processor = this.getProcessor();
        processor.finish(this.processedVec);
        const result = this.processedVec.array.slice();
        // console.debug(`AudioProcessorWorker: Finished into ${result.length} output samples`);
        this.sendResult(result, true);
    }

    private getProcessor(): MultiPitchShifter {
        if (this.processor && !this.paramsAreDirty) {
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
        this.paramsAreDirty = false;

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

function init() {
    const workerImpl = new AudioProcessorWorker();
    // Start manager event loop
    onmessage = (event: MessageEvent<AudioProcessorRequest>) => {
        const message = event.data;
        switch (message.type) {
            case 'init':
                workerImpl.init();
                break;

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
