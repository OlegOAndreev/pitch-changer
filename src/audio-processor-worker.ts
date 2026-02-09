// This is a worker for AudioProcessor

import * as Comlink from 'comlink';

import initWasmModule, { Float32Vec, get_settings, MultiPitchShifter, MultiTimeStretcher, PitchShiftParams, TimeStretchParams } from '../wasm/build/wasm_main_module';
import type { WorkerApi, WorkerParams } from './audio-processor';
import { Float32RingBuffer, pushAllRingBuffer } from './sync';
import type { InterleavedAudio } from './types';

type Processor = MultiPitchShifter | MultiTimeStretcher;

class WorkerImpl implements WorkerApi {
    params: WorkerParams = { processingMode: 'pitch', pitchValue: 1.0 }
    paramsDirty: boolean = false;
    processor: Processor | null = null;
    processorSampleRate: number | null = null;
    processorNumChannels: number | null = null;
    wasmMemory: WebAssembly.Memory | null = null

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

    async process(source: InterleavedAudio, outputSab: SharedArrayBuffer): Promise<void> {
        const output = new Float32RingBuffer(outputSab);
        const chunkSize = output.capacity / 4;

        // Clean the state if previous process() was aborted.
        this.getProcessor(source.sampleRate, source.numChannels).reset();

        let sourcePos = 0;
        // Main processing loop.
        const sourceVec = new Float32Vec(0);
        const processedVec = new Float32Vec(0);
        try {
            while (sourcePos < source.data.length) {
                const nextChunkSize = Math.min(chunkSize, source.data.length - sourcePos);
                sourceVec.set(source.data.subarray(sourcePos, sourcePos + nextChunkSize));
                sourcePos += nextChunkSize;
                const processor = this.getProcessor(source.sampleRate, source.numChannels);
                processedVec.clear();
                processor.process(sourceVec, processedVec);
                // Copy processedVec: if WASM memory gets resized during pushAllRingBuffer, we will get an error with
                // detached ArrayBuffer. 
                const processedArr = new Float32Array(processedVec.array);

                // console.debug(`Processing chunks of size ${sourceVec.len}, num channels ${numChannels} with parameters ${this.params.processingMode}, ${this.params.pitchValue}, got ${processedVec.len}`);
                await pushAllRingBuffer(processedArr, output);
                if (output.isClosed) {
                    return;
                }
            }

            // Finish processing (get remaining data from processor)
            const processor = this.getProcessor(source.sampleRate, source.numChannels);
            processedVec.clear();
            processor.finish(processedVec);
            const processedArr = new Float32Array(processedVec.array);
            // console.debug(`Processing final chunk num channels ${numChannels} with parameters ${this.params.processingMode}, ${this.params.pitchValue}, got ${processedVec.len}`);
            await pushAllRingBuffer(processedArr, output);
        } finally {
            sourceVec.free();
            processedVec.free();
            // Always close output so that the other side does not hang forever.
            output.close();
        }
        console.log(`Wasm memory size: ${this.wasmMemory!.buffer.byteLength}`);
    }

    private getProcessor(sampleRate: number, numChannels: number): Processor {
        if (this.processor && !this.paramsDirty
            && this.processorSampleRate == sampleRate && this.processorNumChannels == numChannels) {
            return this.processor;
        }

        console.log('Recreating processor with params', this.params, 'numChannels', numChannels);
        if (this.processor) {
            this.processor.free();
        }
        try {
            if (this.params.processingMode === 'pitch') {
                const params = new PitchShiftParams(sampleRate, this.params.pitchValue);
                try {
                    this.processor = new MultiPitchShifter(params, numChannels);
                    console.log(`Created MultiPitchShifter with ${params.to_debug_string()}, numChannels ${numChannels}`);
                } finally {
                    params.free();
                }
            } else {
                const params = new TimeStretchParams(sampleRate, this.params.pitchValue);
                try {
                    this.processor = new MultiTimeStretcher(params, numChannels);
                    console.log(`Created MultiTimeStretcher with ${params.to_debug_string()}, numChannels ${numChannels}`);
                } finally {
                    params.free();
                }
            }
            return this.processor!;
        } finally {
            this.paramsDirty = false;
            this.processorSampleRate = sampleRate;
            this.processorNumChannels = numChannels;
        }
    }
}


Comlink.expose(new WorkerImpl());
