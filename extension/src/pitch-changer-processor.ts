import { AudioProcessorClient } from '../../src/audio-processor-client.js';
import { fftSizeForSampleRate } from '../../src/common-utils.js';
import { SamplesQueue } from '../../src/samples-queue.js';
import {
    PROCESSOR_NAME,
    type ProcessingMode,
    type ProcessorOptions,
    type ProcessorRequest,
    type ProcessorSetParams,
    type ProcessorStats,
} from './common.js';

// How many seconds to wait before switching on passthrough optimization.
const PASSTHROUGH_AFTER_SEC = 3;

// How many seconds to wait before sending next stats.
const STATS_MESSAGE_EVERY_SEC = 1;

// PitchProcessor sends data to a separate AudioProcessorWorker and receives the result. It maintains a fixed latency
// ('normal' = fft size * 1.5, 'high' = fft size * 3 for ~70msec and ~150msec latency respectively).
//
// The minimum latency possible is equal to fftSize: the pitch shifting algorithm requires that at least fftSize samples
// must be processed before returning the first results. The output quantum is fftSize/4 (the hop size), but we want
// to account for scheduling hitches.
//
// Optimization: if the input has been only zeros for a few consecutive samples, all the internal buffers are guaranteed
// to contain only zeros. In this case we switch to the fast path: the output is zeroed without any processing at all.
// This optimization is especially important because once started the processor never stops: there is currently no way
// to declare that the processor should be run only if non-zero values are passed to it.
class PitchChangerProcessor extends AudioWorkletProcessor {
    // The worklet node is created with explicit channel count, so WebAudio up/downmixes the input into exactly
    // numChannels channels.
    private readonly numChannels: number;
    private readonly queue: SamplesQueue;
    private client: AudioProcessorClient | null = null;
    private prevProcessingMode: ProcessingMode | null = null;

    // requiredLatency is in samples
    private requiredLatency = 0;
    // currentLatency is the difference between number of input and output 'cursor': number of samples received from
    // input minus number of samples sent to output.
    private currentLatency = 0;

    private numUnderruns = 0;

    private passthroughEnabled = false;
    // We do not immediately switch to passthrough mode: unlike the zero-only fast path, switching to passthrough fast
    // path and back results in an audible click.
    private passthroughRunSamples = 0;
    // After we enable passthrough or zero-only fast path, we need to reset worker state and flush the queue contents.
    private resetRequired = false;

    // Number of consecutive zero input samples, used to switch to the zero-only fast path.
    private zeroRunSamples = 0;
    // The threshold which the zeroRunSamples must exceed in order to switch to the zero-only fast path.
    private zeroPathThreshold = 0;

    private fastPathActive = true;
    // Number of input samples received since the last stats message was sent.
    private samplesSinceStatsMessage = 0;
    // Do not send messages if we are in the zero fast path and have already reported that we are in the fast path.
    private fastPathStatsSent = false;

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
                throw new Error(`PitchChangerProcessor: Unknown message type: ${JSON.stringify(message)}`);
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
        if (params.processingMode === 'formant-preserving-pitch') {
            // Envelope shifting takes another fftSize latency: the internal buffer of the envelope shifter.
            this.requiredLatency += fftSize;
        }
        if (this.prevProcessingMode && this.prevProcessingMode !== params.processingMode) {
            // Flush the queues if we changed the processing mode, otherwise we get the incorrect latency.
            this.resetRequired = true;
        }
        this.prevProcessingMode = params.processingMode;
        this.passthroughEnabled = params.enablePassthroughOptimization && params.pitchValue == 1.0;
        if (this.passthroughEnabled && this.currentLatency === 0) {
            // Hack: if we are just setting params, immediately enable passthrough so that we do not get a click on new
            // pages with pitch value = 1.0.
            this.passthroughRunSamples = PASSTHROUGH_AFTER_SEC * sampleRate;
        }
        // The threshold must be large enough that all non-zero data is processed by PitchShifter. This depends on pitch
        // value: the output_accum_buf in TimeStretcher is shifted by syn_hop_size, which depends not only on fft_size,
        // but also on time_stretch (which equals pitch_shift). Shifting the full fft_size "out of" output_accum_buf
        // takes fft_size / pitch_shift samples.
        this.zeroPathThreshold = fftSize * 4;
    }

    process(inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
        // numberOfInputs and numberOfOutputs are 1 by default, WebAudio does mixing of multiple connected nodes itself.
        const input = inputs[0];
        const output = outputs[0];
        if (!input || input.length === 0) {
            this.fillZeros(output);
            return true;
        }

        // The client has not been created, work as pass-through.
        if (!this.client) {
            this.copySamples(output, input);
            return true;
        }

        const inputLen = input[0].length;

        // Fast path: do a direct copy if the pitch value = 1.0.
        if (this.passthroughEnabled) {
            this.passthroughRunSamples += inputLen;
            if (this.passthroughRunSamples >= PASSTHROUGH_AFTER_SEC * sampleRate) {
                this.copySamples(output, input);
                // When the pitch value changes, clear both the client and the queue from old data.
                this.resetRequired = true;
                this.fastPathActive = true;
                this.maybeSendStats(inputLen);
                return true;
            }
        } else {
            this.passthroughRunSamples = 0;
        }

        // Fast path: do a direct zeroing if the input is zero.
        if (this.isAllZeros(input)) {
            this.zeroRunSamples += inputLen;
            if (this.zeroRunSamples >= this.zeroPathThreshold) {
                // Do not touch queue or currentLatency: the queue should contain only zeros and will be used as soon as
                // we receive the first non-zero input sample. The async part of filling the queue does not matter as
                // well: essentially we reorder zeros with zeros.
                this.fillZeros(output);
                this.fastPathActive = true;
                this.maybeSendStats(inputLen);
                return true;
            }
        } else {
            this.zeroRunSamples = 0;
        }

        // Do a reset+flush after returning from passthrough fast path.
        if (this.resetRequired) {
            this.client.reset();
            this.queue.skip(this.queue.length);
            // This requires that client is reset beforehand.
            this.currentLatency = 0;
            this.resetRequired = false;
        }

        this.sendInput(input);
        this.writeOutput(output);
        this.fastPathActive = false;
        this.maybeSendStats(inputLen);
        return true;
    }

    private copySamples(output: Float32Array[], input: Float32Array[]) {
        const numChannels = Math.min(input.length, this.numChannels);
        for (let ch = 0; ch < numChannels; ch++) {
            output[ch].set(input[ch], 0);
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

        const written = this.queue.popNonInterleaved(output);
        this.currentLatency -= written;
        if (written < output[0].length) {
            this.numUnderruns++;
        }
    }

    private isAllZeros(inputChannels: Float32Array[]): boolean {
        for (const channel of inputChannels) {
            for (let i = 0; i < channel.length; i++) {
                if (channel[i] !== 0.0) {
                    return false;
                }
            }
        }
        return true;
    }

    private fillZeros(outputChannels: Float32Array[]): void {
        for (const channel of outputChannels) {
            channel.fill(0.0);
        }
    }

    private maybeSendStats(numSamples: number): void {
        if (this.fastPathActive && this.fastPathStatsSent) {
            return;
        }

        this.samplesSinceStatsMessage += numSamples;
        if (this.samplesSinceStatsMessage < STATS_MESSAGE_EVERY_SEC * sampleRate) {
            return;
        }

        this.samplesSinceStatsMessage = 0;
        this.port.postMessage({
            type: 'pitch-changer-extension-processor-stats',
            fastPathActive: this.fastPathActive,
            // The queue and latency are stale/meaningless when the fast path is active, report them as zero.
            currentLatencyMs: this.fastPathActive ? 0 : (this.currentLatency / sampleRate) * 1000,
            numUnderruns: this.numUnderruns,
            queueLength: this.fastPathActive ? 0 : this.queue.length,
        } as ProcessorStats);
        this.fastPathStatsSent = this.fastPathActive;
    }
}

registerProcessor(PROCESSOR_NAME, PitchChangerProcessor);
