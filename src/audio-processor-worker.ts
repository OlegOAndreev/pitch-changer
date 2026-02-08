// This is a worker for AudioProcessor

import * as Comlink from 'comlink';

import initWasmModule, { Float32Vec, PitchShifter, PitchShiftParams, TimeStretcher, TimeStretchParams } from '../wasm/build/wasm_main_module';
import type { WorkerParams, WorkerApi } from './audio-processor';
import { Float32RingBuffer } from './ring-buffer';

// Write in chunks of 16384 samples ~= 350-400msec at regular sample rates
const CHUNK_SIZE = 16384;

type Processor = PitchShifter | TimeStretcher;

class WorkerImpl implements WorkerApi {
    params: WorkerParams = { processingMode: 'pitch', pitchValue: 1.0 }
    paramsDirty: boolean = false;
    processor: Processor | null = null;
    wasmMemory: WebAssembly.Memory | null = null

    async init(): Promise<boolean> {
        // Initialize WASM module. It must be done in the function, not top-level:
        // https://github.com/GoogleChromeLabs/comlink/issues/635 
        const module = await initWasmModule();
        this.wasmMemory = module.memory;
        console.log(`Initial wasm memory size: ${this.wasmMemory.buffer.byteLength}`);
        return true;
    }

    setParams(newParams: WorkerParams): void {
        this.params = newParams;
        this.paramsDirty = true;
    }

    async process(source: Float32Array, sampleRate: number, outputSab: SharedArrayBuffer): Promise<void> {
        const output = new Float32RingBuffer(outputSab);
        // Force the higher buffer size to prevent underruns. We also want to be sure that the chunks returned from the
        // processor do not exceed the ring buffer capacity.
        if (output.capacity < CHUNK_SIZE * 8) {
            throw new Error(`Output buffer is too small, should be at least ${CHUNK_SIZE * 8}`)
        }

        let sourcePos = 0;
        // Main processing loop.
        const sourceVec = new Float32Vec(0);
        const processedVec = new Float32Vec(0);
        try {
            while (sourcePos < source.length) {
                const chunkSize = Math.min(CHUNK_SIZE, source.length - sourcePos);
                sourceVec.set(source.subarray(sourcePos, sourcePos + chunkSize));
                sourcePos += chunkSize;
                const processor = this.getProcessor(sampleRate);
                processedVec.clear();
                processor.process(sourceVec, processedVec);

                await output.waitPushAsync(processedVec.len);
                const n = output.push(processedVec.array);
                if (output.isClosed) {
                    return;
                }
                // console.debug(`Processing chunks of size ${sourceVec.len} with parameters ${this.params.processingMode}, ${this.params.pitchValue}, got ${processedVec.len}, stored ${n} samples`);
                if (n !== processedVec.len && !output.isClosed) {
                    throw new Error(`Internal error: unexpected push(${processedVec.len}) result: ${n}`);
                }
            }

            // Finish processing (get remaining data from processor)
            const processor = this.getProcessor(sampleRate);
            processedVec.clear();
            processor.finish(processedVec);
            await output.waitPushAsync(processedVec.len);
            const n = output.push(processedVec.array);
            // console.debug(`Processing final chunk with parameters ${this.params.processingMode}, ${this.params.pitchValue}, got ${processedVec.len}, stored ${n} samples`);
            if (output.isClosed) {
                return;
            }
            if (n !== processedVec.len && !output.isClosed) {
                throw new Error(`Internal error: unexpected push(${processedVec.len}) result: ${n}`);
            }
        } finally {
            sourceVec.free();
            processedVec.free();
        }
        console.log(`Wasm memory size: ${this.wasmMemory!.buffer.byteLength}`);
    }

    private getProcessor(sampleRate: number): Processor {
        if (this.processor && !this.paramsDirty) {
            return this.processor;
        }

        console.log('Recreating processor with params', this.params);
        if (this.processor) {
            this.processor.free();
        }
        try {
            if (this.params.processingMode === 'pitch') {
                const params = new PitchShiftParams(sampleRate, this.params.pitchValue);
                try {
                    this.processor = new PitchShifter(params);
                    console.log(`Created PitchShifter with ${params.to_debug_string()}`);
                } finally {
                    params.free();
                }
            } else {
                const params = new TimeStretchParams(sampleRate, this.params.pitchValue);
                try {
                    this.processor = new TimeStretcher(params);
                    console.log(`Created TimeStretcher with ${params.to_debug_string()}`);
                } finally {
                    params.free();
                }
            }
            return this.processor;
        } finally {
            this.paramsDirty = false;
        }

    }
}


Comlink.expose(new WorkerImpl());
