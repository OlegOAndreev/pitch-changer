// This is a worker code for player.ts

import initWasmModule, {
    Float32Vec,
    get_settings,
    MultiPitchShifter,
    PitchShiftParams,
} from '../wasm/build/wasm_main_module';
import type { ProcessingMode } from './types';

// Message types for communication with the player worker
export interface WorkerParams {
    processingMode: ProcessingMode;
    pitchValue: number;
    sampleRate: number;
    numChannels: number;
    fftSize: number;
}

export interface ProcessSamplesMessage {
    type: 'processSamples';
    samples: Float32Array;
}

export interface SetParamsMessage {
    type: 'setParams';
    params: WorkerParams;
}

export interface FinishMessage {
    type: 'finish';
}

export interface ProcessedSamplesMessage {
    type: 'processedSamples';
    samples: Float32Array;
    finished: boolean;
}

export type WorkerRequestMessage = SetParamsMessage | ProcessSamplesMessage | FinishMessage;

export type WorkerResponseMessage = ProcessedSamplesMessage;

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

    // Initialize WASM module
    async init(): Promise<void> {
        const module = await initWasmModule();
        const wasmMemory = module.memory;
        console.debug(`Player worker: Wasm settings: ${get_settings()}`);
        console.debug(`Player worker: Initial wasm memory size: ${wasmMemory.buffer.byteLength}`);
    }

    setParams(newParams: WorkerParams): void {
        this.params = newParams;
        this.paramsDirty = true;
    }

    processSamples(samples: Float32Array): Float32Array {
        const processor = this.getProcessor();
        this.inputVec.set(samples);
        this.processedVec.clear();
        processor.process(this.inputVec, this.processedVec);
        // Clone the array: we will transfer the buffer back and allow processedVec to be reused.
        const result = this.processedVec.array.slice();
        // console.debug(
        //     `AudioProcessorWorker: Processed ${samples.length} input samples into ${result.length} output samples`,
        // );
        return result;
    }

    finish(): Float32Array {
        const processor = this.getProcessor();
        this.processedVec.clear();
        processor.finish(this.processedVec);
        const result = this.processedVec.array.slice();
        console.debug(`AudioProcessorWorker: Finished into ${result.length} output samples`);
        return result;
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
}

const workerImpl = new AudioProcessorWorker();
workerImpl.init();

addEventListener('message', async (event: MessageEvent<WorkerRequestMessage>) => {
    const message = event.data;
    switch (message.type) {
        case 'setParams':
            workerImpl.setParams(message.params);
            break;

        case 'processSamples': {
            try {
                const samples = workerImpl.processSamples(message.samples);
                const response: ProcessedSamplesMessage = {
                    type: 'processedSamples',
                    samples,
                    finished: false,
                };
                self.postMessage(response, { transfer: [samples.buffer] });
            } catch (error) {
                console.error('AudioProcessorWorker: failed to process samples:', error);
            }
            break;
        }

        case 'finish': {
            try {
                const samples = workerImpl.finish();
                const response: ProcessedSamplesMessage = {
                    type: 'processedSamples',
                    samples,
                    finished: true,
                };
                self.postMessage(response, { transfer: [samples.buffer] });
            } catch (error) {
                console.error('AudioProcessorWorker: failed to process samples:', error);
            }
            break;
        }

        default:
            console.error('Player worker: Unknown message type:', message);
            break;
    }
});
