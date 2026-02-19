import { Float32Vec, MultiPitchShifter, PitchShiftParams } from '../wasm/build/wasm_main_module';
import type { InterleavedAudio, ProcessingMode } from './types';

export interface BenchmarkResults {
    pitch: number;
    'formant-preserving-pitch': number;
    time: number;
}

// Generate random noise with uniform distribution between -1 and 1
function generateNoise(samplesPerChannel: number, numChannels: number): Float32Array {
    const totalSamples = samplesPerChannel * numChannels;
    const arr = new Float32Array(totalSamples);
    for (let i = 0; i < totalSamples; i++) {
        arr[i] = (Math.random() - 0.5) * 2;
    }
    return arr;
}

// Process audio using the same logic as audio-processor-worker.ts but in the main thread
function processAudio(input: InterleavedAudio, processingMode: ProcessingMode, pitchValue: number) {
    const sampleRate = input.sampleRate;
    const numChannels = input.numChannels;
    let processor: MultiPitchShifter;

    // Calculate parameters based on processing mode
    let pitchShift = pitchValue;
    let timeStretch = 1.0;
    if (processingMode === 'time') {
        // For time stretch mode, use pitchValue as time_stretch and keep pitch at 1.0
        pitchShift = 1.0;
        timeStretch = pitchValue;
    }

    const params = new PitchShiftParams(sampleRate, pitchShift, timeStretch);
    if (processingMode === 'formant-preserving-pitch') {
        params.quefrency_cutoff = 1.0;
    }
    try {
        processor = new MultiPitchShifter(params, numChannels);
    } finally {
        params.free();
    }

    try {
        const CHUNK_SAMPLES = 65536;
        let inputPos = 0;
        const inputVec = new Float32Vec(0);
        const outputVec = new Float32Vec(0);
        try {
            while (inputPos < input.data.length) {
                const nextChunkSize = Math.min(CHUNK_SAMPLES, input.data.length - inputPos);
                inputVec.set(input.data.subarray(inputPos, inputPos + nextChunkSize));
                inputPos += nextChunkSize;

                outputVec.clear();
                processor.process(inputVec, outputVec);
            }
            outputVec.clear();
            processor.finish(outputVec);
        } finally {
            inputVec.free();
            outputVec.free();
        }
    } finally {
        processor.free();
    }
}

export function runBenchmark(
    sampleRate: number,
    numChannels: number,
    durationSeconds: number,
    pitchValue: number,
): BenchmarkResults {
    const samplesPerChannel = sampleRate * durationSeconds;
    const noiseData = generateNoise(samplesPerChannel, numChannels);
    const input: InterleavedAudio = {
        data: noiseData,
        sampleRate,
        numChannels,
    };

    const modes: ProcessingMode[] = ['pitch', 'formant-preserving-pitch', 'time'];
    const results: Partial<BenchmarkResults> = {};

    for (const mode of modes) {
        const startTime = performance.now();
        processAudio(input, mode, pitchValue);
        const endTime = performance.now();
        const processingTimeMs = endTime - startTime;

        const ratio = (durationSeconds * 1000) / processingTimeMs;
        results[mode as keyof BenchmarkResults] = ratio;
    }

    return results as BenchmarkResults;
}
