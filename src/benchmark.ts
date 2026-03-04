import { Float32Vec, MultiPitchShifter, PitchShiftParams } from '../wasm/build/wasm_main_module';
import type { InterleavedAudio, ProcessingMode } from './types';

export interface BenchmarkResults {
    pitch: number;
    'formant-preserving-pitch': number;
    time: number;
}

// Simple deterministic pseudo-random number generator using LCG
class SeededRandom {
    private seed: number;

    constructor(seed: number) {
        this.seed = seed;
    }

    next(): number {
        // Simple LCG: using parameters from glibc
        this.seed = (1103515245 * this.seed + 12345) & 0x7fffffff;
        return this.seed / 0x7fffffff;
    }
}

// Generate random noise with uniform distribution between -1 and 1
function generateNoise(samplesPerChannel: number, numChannels: number): Float32Array {
    const totalSamples = samplesPerChannel * numChannels;
    const arr = new Float32Array(totalSamples);
    const randomGen = new SeededRandom(42);
    for (let i = 0; i < totalSamples; i++) {
        arr[i] = (randomGen.next() - 0.5) * 2;
    }
    return arr;
}

// Generate several sine waves with different frequencies and add into one Float32Array
function generateSineWaves(
    samplesPerChannel: number,
    numChannels: number,
    sampleRate: number,
    freqs: number[],
): Float32Array {
    const totalSamples = samplesPerChannel * numChannels;
    const arr = new Float32Array(totalSamples);

    for (const freq of freqs) {
        const step = (2 * Math.PI * freq) / sampleRate;
        for (let idx = 0; idx < samplesPerChannel; idx++) {
            const sineValue = Math.sin(step * idx);
            for (let channel = 0; channel < numChannels; channel++) {
                const index = idx * numChannels + channel;
                arr[index] += sineValue;
            }
        }
    }

    // Normalize to [-1, 1]
    let maxAbsValue = 0;
    for (let i = 0; i < totalSamples; i++) {
        const absValue = Math.abs(arr[i]);
        if (absValue > maxAbsValue) {
            maxAbsValue = absValue;
        }
    }
    if (maxAbsValue > 0) {
        const scale = 1.0 / maxAbsValue;
        for (let i = 0; i < totalSamples; i++) {
            arr[i] *= scale;
        }
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

function measureTestTime(sampleRate: number, numChannels: number, pitchValue: number): number {
    // Test how much time an iteration takes.
    const testSamples = generateNoise(sampleRate * 10, numChannels);
    const testData: InterleavedAudio = {
        data: testSamples,
        sampleRate,
        numChannels,
    };
    const testStartTime = performance.now();
    processAudio(testData, 'pitch', pitchValue);
    return performance.now() - testStartTime;
}

export function runBenchmark(
    sampleRate: number,
    numChannels: number,
    pitchValue: number,
    withNoise: boolean,
): BenchmarkResults {
    const NUM_ITERATIONS = 4;
    const timePer10Sec = measureTestTime(sampleRate, numChannels, pitchValue);
    // We want the whole test to take ~5 sec.
    const durationSeconds = Math.round((1000 * 10) / (timePer10Sec * NUM_ITERATIONS));
    console.log(`Generating benchmark samples for ${durationSeconds.toFixed(1)}sec`);

    const samplesPerChannel = sampleRate * durationSeconds;
    let inputSamples;
    if (withNoise) {
        inputSamples = generateNoise(samplesPerChannel, numChannels);
    } else {
        const freqs = [];
        const randomGen = new SeededRandom(42);
        for (let i = 0; i < 50; i++) {
            freqs.push(randomGen.next() * 10000);
        }
        inputSamples = generateSineWaves(samplesPerChannel, numChannels, sampleRate, freqs);
    }
    const inputData: InterleavedAudio = {
        data: inputSamples,
        sampleRate,
        numChannels,
    };

    const modes: ProcessingMode[] = ['time', 'pitch', 'formant-preserving-pitch'];
    const results: BenchmarkResults = {
        pitch: 0.0,
        'formant-preserving-pitch': 0.0,
        time: 0.0,
    };

    for (let iter = 0; iter < NUM_ITERATIONS; iter++) {
        for (const mode of modes) {
            const startTime = performance.now();
            processAudio(inputData, mode, pitchValue);
            const endTime = performance.now();
            const processingTimeMs = endTime - startTime;

            const ratio = (durationSeconds * 1000) / processingTimeMs;
            results[mode] = Math.max(ratio, results[mode]);
        }
    }

    return results as BenchmarkResults;
}
