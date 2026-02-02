import { PitchShifter, PitchShiftParams, TimeStretcher, TimeStretchParams } from '../wasm/build/wasm_main_module';

export function shiftPitch(audioData: Float32Array, sampleRate: number, pitchRatio: number): Float32Array {
    const params = new PitchShiftParams(sampleRate, pitchRatio);
    try {
        const shifter = new PitchShifter(params);
        try  {
            const chunk1 = shifter.process(audioData);
            const chunk2 = shifter.finish();
            // Concatenate chunk1 and chunk2
            const result = new Float32Array(chunk1.length + chunk2.length);
            result.set(chunk1, 0);
            result.set(chunk2, chunk1.length);
            return result;
        } finally {
            shifter.free();
        }
    } finally {
        params.free();
    }
}

export function timeStretch(audioData: Float32Array, sampleRate: number, stretchRatio: number): Float32Array {
    const params = new TimeStretchParams(sampleRate, stretchRatio);
    try {
        const stretcher = new TimeStretcher(params);
        try {
            const chunk1 = stretcher.process(audioData);
            const chunk2 = stretcher.finish();
            // Concatenate chunk1 and chunk2
            const result = new Float32Array(chunk1.length + chunk2.length);
            result.set(chunk1, 0);
            result.set(chunk2, chunk1.length);
            return result;
        } finally {
            stretcher.free();
        }
    } finally {
        params.free();
    }
}
