// This is a worker for AudioProcessor

import * as Comlink from 'comlink';

import initWasmModule, { PitchShifter, PitchShiftParams, TimeStretcher, TimeStretchParams } from '../wasm/build/wasm_main_module';
import type { WorkerParams, WorkerApi } from './audio-processor';
import { Float32RingBuffer } from './ring-buffer';

// Write in chunks of 2048 samples ~= 40-50msec at normal sample rates
const CHUNK_SIZE = 2048;

type Processor = PitchShifter | TimeStretcher;

class WorkerImpl implements WorkerApi {
    params: WorkerParams = { processingMode: 'pitch', pitchValue: 1.0 }
    paramsDirty: boolean = false;
    processor: Processor | null = null;

    async init(): Promise<boolean> {
        // Initialize WASM module. It must be done in the function, not top-level:
        // https://github.com/GoogleChromeLabs/comlink/issues/635 
        await initWasmModule();
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
        while (sourcePos < source.length) {
            const chunkSize = Math.min(CHUNK_SIZE, source.length - sourcePos);
            const chunk = source.subarray(sourcePos, sourcePos + chunkSize);
            sourcePos += chunkSize;
            const processor = this.getProcessor(sampleRate);
            const processedChunk = processor.process(chunk);

            await output.waitPushAsync(processedChunk.length);
            const n = output.push(processedChunk);
            if (output.isClosed) {
                return;
            }
            // console.debug(`Processing chunks of size ${chunk.length} with parameters ${this.params.processingMode}, ${this.params.pitchValue}, got ${processedChunk.length}, stored ${n} samples`);
            if (n !== processedChunk.length && !output.isClosed) {
                throw new Error(`Internal error: unexpected push(${processedChunk.length}) result: ${n}`);
            }
        }

        // Finish processing (get remaining data from processor)
        const processor = this.getProcessor(sampleRate);
        const finalChunk = processor.finish();
        await output.waitPushAsync(finalChunk.length);
        const n = output.push(finalChunk);
        // console.debug(`Processing final chunk with parameters ${this.params.processingMode}, ${this.params.pitchValue}, got ${finalChunk.length}, stored ${n} samples`);
        if (output.isClosed) {
            return;
        }
        if (n !== finalChunk.length && !output.isClosed) {
            throw new Error(`Internal error: unexpected push(${finalChunk.length}) result: ${n}`);
        }
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
                } finally {
                    params.free();
                }
            } else {
                const params = new TimeStretchParams(sampleRate, this.params.pitchValue);
                try {
                    this.processor = new TimeStretcher(params);
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
