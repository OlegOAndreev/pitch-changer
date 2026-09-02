import { AudioProcessorClient } from '../../src/audio-processor-client.js';
import { fftSizeForSampleRate } from '../../src/common-utils.js';
import { SamplesQueue } from '../../src/samples-queue.js';
import { PROCESSOR_NAME, type ProcessorOptions, type ProcessorRequest, type ProcessorSetParams } from './common.js';

// PitchProcessor sends data to a separate AudioProcessorWorker and receives the result. It maintains a fixed latency
// ('normal' = fft size * 1.5, 'high' = fft size * 3 for ~70msec and ~150msec latency respectively).
//
// The minimum latency possible is equal to fftSize: the pitch shifting algorithm requires that at least fftSize samples
// must be processed before returning the first results. The output quantum is fftSize/4 (the hop size), but we want
// to account for scheduling hitches.
class PitchChangerProcessor extends AudioWorkletProcessor {
    // The worklet node is created with explicit channel count, so WebAudio up/downmixes the input into exactly
    // numChannels channels. Unlike the main app, we set all parameters from inside the worklet processor, which require
    private readonly numChannels: number;
    private readonly queue: SamplesQueue;
    private client: AudioProcessorClient | null = null;

    // requiredLatency is in samples
    private requiredLatency = 0;
    // currentLatency is the difference between number of input and output 'cursor': number of samples received from
    // input minus number of samples sent to output.
    private currentLatency = 0;

    private numUnderruns = 0;

    constructor(options: AudioWorkletNodeOptions) {
        super();

        const processorOptions = options.processorOptions as ProcessorOptions;
        this.numChannels = processorOptions.numChannels;
        this.queue = new SamplesQueue(this.numChannels);

        this.port.onmessage = (event: MessageEvent<ProcessorRequest>) => {
            this.onMessage(event.data);
        };
    }

    private onMessage(message: ProcessorRequest) {
        switch (message.type) {
            case 'pitch-changer-extension-processor-init':
                this.client = new AudioProcessorClient(
                    message.audioProcessorClientPort,
                    this.onProcessedSamples.bind(this),
                );
                break;

            case 'pitch-changer-extension-processor-set-params':
                this.setParams(message);
                break;

            default:
                throw new Error(`PlayerProcessor: Unknown message type: ${JSON.stringify(message)}`);
        }
    }

    setParams(params: ProcessorSetParams) {
        // Use the smaller sampling rate to make latency lower.
        const fftSize = fftSizeForSampleRate(sampleRate, 40);
        this.client!.setParams(params.processingMode, params.pitchValue, sampleRate, this.numChannels, fftSize);
        switch (params.targetLatency) {
            case 'normal':
                this.requiredLatency = fftSize + fftSize / 2;
                break;
            case 'high':
                this.requiredLatency = fftSize * 3;
                break;
        }
    }

    process(inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
        // numberOfInputs and numberOfOutputs are 1 by default, WebAudio does mixing of multiple connected nodes itself.
        const input = inputs[0];
        const output = outputs[0];
        if (!input || input.length === 0) {
            this.fillZeros(output);
            return true;
        }

        // TODO: Add fast path if there have been fftSize*2 consecutive zeros in input.

        // // We want to skip processing inputs if the input is zero.
        // if (this.isAllZeros(input)) {
        //     // this.fillZeros(output);
        //     return true;
        // }

        // The client has not been created, work as pass-through.
        if (!this.client) {
            this.copySamples(output, input);
            return true;
        }

        this.sendInput(input);
        this.writeOutput(output);
        return true;
    }

    private copySamples(output: Float32Array[], input: Float32Array[]) {
        const numChannels = Math.min(input.length, this.numChannels);
        const blockSize = output[0].length;
        for (let ch = 0; ch < numChannels; ch++) {
            const inputChannel = input[ch];
            const outputChannel = output[ch];
            for (let i = 0; i < blockSize; i++) {
                outputChannel[i] = inputChannel[i] * 2.0;
            }
        }
    }

    private sendInput(input: Float32Array[]): void {
        const inputLen = input[0].length;
        this.currentLatency += inputLen;
        // Convert planar to interleaved format. The intermediate buffer is transferred to worker, no need to cache it.
        const numChannels = this.numChannels;
        const buffer = new Float32Array(inputLen * numChannels);
        for (let ch = 0; ch < numChannels; ch++) {
            const channelData = input[ch];
            for (let i = 0; i < inputLen; i++) {
                buffer[i * numChannels + ch] = channelData[i];
            }
        }
        this.client!.processSamples(buffer);
    }

    // We never close the client, finished is always false.
    private onProcessedSamples(samples: Float32Array, _finished: boolean): void {
        this.queue.pushInterleaved(samples);
    }

    private writeOutput(output: Float32Array[]): void {
        if (this.currentLatency < this.requiredLatency) {
            // This happens either at the start of processing, or after changing latency to a higher one. We assume that
            // output quantum always divides latency.
            this.fillZeros(output);
            return;
        } else if (this.currentLatency > this.requiredLatency) {
            // Skip the queued elements which are already late.
            const skipped = this.queue.skip(this.currentLatency - this.requiredLatency);
            this.currentLatency -= skipped;
            // Check if the queue became empty after skipping,
            if (this.currentLatency > this.requiredLatency) {
                this.fillZeros(output);
                return;
            }
        }

        const outputLen = output[0].length;
        const written = this.queue.popNonInterleaved(output);
        this.currentLatency -= written;
        const underrun = outputLen - written;
        if (underrun > 0) {
            this.numUnderruns++;
            // TODO: Report underruns via messages once in a while?
            console.error("Got underrun ${underrun}");
        }
    }

    // private isAllZeros(inputChannels: Float32Array[]): boolean {
    //     for (const channel of inputChannels) {
    //         for (let i = 0; i < channel.length; i++) {
    //             if (channel[i] !== 0.0) {
    //                 return false;
    //             }
    //         }
    //     }
    //     return true;
    // }

    private fillZeros(outputChannels: Float32Array[]): void {
        for (const channel of outputChannels) {
            channel.fill(0.0);
        }
    }
}

registerProcessor(PROCESSOR_NAME, PitchChangerProcessor);
