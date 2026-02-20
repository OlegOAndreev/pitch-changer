// This is a worker for AudioProcessor

import * as Comlink from 'comlink';

import initWasmModule, {
    Float32Vec,
    get_settings,
    MultiPitchShifter,
    PitchShiftParams,
} from '../wasm/build/wasm_main_module';
import type { WorkerApi, WorkerParams } from './audio-processor';
import { Float32RingBuffer, pushAllRingBuffer } from './sync';
import type { InterleavedAudio, ProcessingMode } from './types';

type Processor = MultiPitchShifter;

class WorkerImpl implements WorkerApi {
    params: WorkerParams = { processingMode: 'pitch', pitchValue: 1.0 };
    paramsDirty: boolean = false;
    currentProcessorMode: ProcessingMode | null = null;
    processor: Processor | null = null;
    processorNumChannels: number | null = null;
    wasmMemory: WebAssembly.Memory | null = null;

    async init(): Promise<boolean> {
        // Initialize WASM module. It must be done in the function, not top-level:
        // https://github.com/GoogleChromeLabs/comlink/issues/635
        const module = await initWasmModule();
        this.wasmMemory = module.memory;
        console.log(`Wasm settings: ${get_settings()}`);
        console.log(`Initial wasm memory size: ${this.wasmMemory.buffer.byteLength}`);
        return true;
    }

    setParams(newParams: WorkerParams): void {
        this.params = newParams;
        this.paramsDirty = true;
    }

    async process(input: InterleavedAudio, outputSab: SharedArrayBuffer): Promise<void> {
        const output = new Float32RingBuffer(outputSab);
        const chunkSize = output.capacity / 4;

        this.paramsDirty = true;
        // Clean the state if previous process() was aborted.
        this.getProcessor(input.sampleRate, input.numChannels).reset();

        let inputPos = 0;
        // Main processing loop.
        const inputVec = new Float32Vec(0);
        const processedVec = new Float32Vec(0);
        try {
            while (inputPos < input.data.length) {
                const nextChunkSize = Math.min(chunkSize, input.data.length - inputPos);
                inputVec.set(input.data.subarray(inputPos, inputPos + nextChunkSize));
                inputPos += nextChunkSize;

                processedVec.clear();
                this.getProcessor(input.sampleRate, input.numChannels).process(inputVec, processedVec);
                // Copy processedVec: if WASM memory gets resized during pushAllRingBuffer, we will get an error with
                // detached ArrayBuffer.
                const processedArr = new Float32Array(processedVec.array);

                // console.debug(`Processing chunks of size ${inputVec.len}, num channels ${input.numChannels} with parameters ${this.params.processingMode}, ${this.params.pitchValue}, got ${processedVec.len}`);
                await pushAllRingBuffer(processedArr, output);
                if (output.isClosed) {
                    return;
                }
            }

            // Finish processing (get remaining data from processor)
            processedVec.clear();
            this.getProcessor(input.sampleRate, input.numChannels).finish(processedVec);
            const processedArr = new Float32Array(processedVec.array);
            // console.debug(`Processing final chunk num channels ${input.numChannels} with parameters ${this.params.processingMode}, ${this.params.pitchValue}, got ${processedVec.len}`);
            await pushAllRingBuffer(processedArr, output);
        } finally {
            inputVec.free();
            processedVec.free();
            // Always close output so that the other side does not hang forever.
            output.close();
        }
        console.log(`Wasm memory size: ${this.wasmMemory!.buffer.byteLength}`);
    }

    private getProcessor(sampleRate: number, numChannels: number): Processor {
        if (this.processor && this.processorNumChannels === numChannels && !this.paramsDirty) {
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
        const params = new PitchShiftParams(sampleRate, pitchShift, timeStretch);
        try {
            if (this.params.processingMode === 'formant-preserving-pitch') {
                params.quefrency_cutoff = 1.0;
            }

            if (this.processor && this.processorNumChannels === numChannels) {
                console.log(`Updating processor parameters to ${params.to_debug_string()}`);
                this.processor.update_params(params);
            } else {
                console.log(`Creating new processor with parameters to ${params.to_debug_string()}`);
                if (this.processor) {
                    this.processor.free();
                }
                this.processor = new MultiPitchShifter(params, numChannels);
            }
        } finally {
            params.free();
        }
        this.processorNumChannels = numChannels;
        this.paramsDirty = false;

        return this.processor!;
    }
}

Comlink.expose(new WorkerImpl());
