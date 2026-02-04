import * as Comlink from 'comlink';

import initWasmModule, { PitchShifter, PitchShiftParams, TimeStretcher, TimeStretchParams } from '../wasm/build/wasm_main_module';
import type { AudioProcessorWorker } from './audio-processor';

try {
    await initWasmModule();
} catch (error) {
    console.error('Failed to initialize WASM module:', error);
}

const api: AudioProcessorWorker = {
    shiftPitch(audioData: Float32Array, sampleRate: number, pitchRatio: number): Float32Array {
        console.log(`Shifting pitch of ${audioData.length} samples by ${pitchRatio}`);
        const startTime = performance.now();
        const params = new PitchShiftParams(sampleRate, pitchRatio);
        try {
            const shifter = new PitchShifter(params);
            try {
                // TODO: Use smaller chunks and check for cancellation
                const chunk1 = shifter.process(audioData);
                const chunk2 = shifter.finish();
                // Concatenate chunk1 and chunk2
                const result = new Float32Array(chunk1.length + chunk2.length);
                result.set(chunk1, 0);
                result.set(chunk2, chunk1.length);
                console.log(`Shifted pitch of ${audioData.length} samples by ${pitchRatio} in ${performance.now() - startTime}ms`);
                return result;
            } finally {
                shifter.free();
            }
        } finally {
            params.free();
        }
    },

    timeStretch(audioData: Float32Array, sampleRate: number, stretchRatio: number): Float32Array {
        console.log(`Stretching time for ${audioData.length} samples by ${stretchRatio}`);
        const startTime = performance.now();
        const params = new TimeStretchParams(sampleRate, stretchRatio);
        try {
            const stretcher = new TimeStretcher(params);
            try {
                // TODO: Use smaller chunks and check for cancellation
                const chunk1 = stretcher.process(audioData);
                const chunk2 = stretcher.finish();
                // Concatenate chunk1 and chunk2
                const result = new Float32Array(chunk1.length + chunk2.length);
                result.set(chunk1, 0);
                result.set(chunk2, chunk1.length);
                console.log(`Stretched time for ${audioData.length} samples by ${stretchRatio} in ${performance.now() - startTime}ms`);
                return result;
            } finally {
                stretcher.free();
            }
        } finally {
            params.free();
        }
    }
};

Comlink.expose(api);
console.log('Audio worker ready with Comlink');
