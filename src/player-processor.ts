// AudioWorklet processor for Player.
import { AudioProcessorClient } from './audio-processor-client';
import { PlayerProcessorQueue } from './player-processor-queue';
import {
    playerProcessorName,
    type LatestSamplesMessage,
    type PlaybackFinishedMessage,
    type PlayerProcessorOptions,
    type PlayerRequest,
} from './player-types';

const CONSOLE_LOG_MIN_DELTA = 100;

class PlayerProcessor extends AudioWorkletProcessor {
    private readonly numChannels: number;
    private readonly bufferSamples: number;

    private input: Float32Array | null = null;
    // Offset in samples from start of input array.
    private inputOffset = 0;
    // True if we sent all input samples to process.
    private inputConsumed = false;

    private client: AudioProcessorClient | null = null;
    // True if we have received our first processed chunk.
    private firstChunkProcessed = false;
    // True if we sent the samples to process and haven't received a result.
    private processingInProgress = false;
    // Chunks of processed samples.
    private processedQueue: PlayerProcessorQueue;
    // True if we got the last processed chunk (this does not mean we have played it already)
    private processingFinished = false;

    private playbackFinished = false;

    private lastConsoleLogTime = 0;

    constructor(options: AudioWorkletNodeOptions) {
        super();

        const processorOptions = options.processorOptions as PlayerProcessorOptions;
        this.numChannels = processorOptions.numChannels;
        this.bufferSamples = processorOptions.bufferSamples;
        this.processedQueue = new PlayerProcessorQueue(this.numChannels);

        this.port.onmessage = (event: MessageEvent<PlayerRequest>) => {
            this.onMessage(event.data);
        };
    }

    process(_inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
        // Fill output if we did not still receive the first processed chunk or finished processing (either due to
        // processing all input or stopping the playback).
        if (!this.firstChunkProcessed || this.playbackFinished) {
            this.fillZeros(outputs[0]);
            return !this.playbackFinished;
        }

        this.writeOutput(outputs[0]);
        // If we are low on processed samples, request processing new samples.
        if (this.processedQueue.length < this.bufferSamples && !this.inputConsumed && !this.processingInProgress) {
            this.requestProcessSamples();
        }

        // If we have consumed all processed samples, send finished message and request termination.
        if (this.processingFinished && this.processedQueue.length === 0) {
            this.reportFinished();
            return false;
        }

        return true;
    }

    private onMessage(message: PlayerRequest) {
        switch (message.type) {
            case 'init':
                this.doInit(message.input, message.clientPort);
                break;
            case 'getLatestSamples': {
                this.sendLatestSamples(message.numSamples);
                break;
            }
            case 'stop':
                this.playbackFinished = true;
                break;
            default:
                console.error('PlayerProcessor: Unknown message type:', message);
                break;
        }
    }

    private doInit(input: Float32Array, clientPort: MessagePort) {
        if (this.input) {
            console.error('Double init() called');
            return;
        }
        this.input = input;
        this.client = new AudioProcessorClient(clientPort, this.onProcessedSamples.bind(this));
        // Reset the AudioProcessor, because the it may have been used by another player.
        this.client.reset();
        // Prefill the buffer, next requestProcessSamples() will be called from process().
        this.requestProcessSamples();
        console.log('PlayerProcessor initialized');
    }

    private fillZeros(outputChannels: Float32Array[]) {
        for (let ch = 0; ch < this.numChannels; ch++) {
            outputChannels[ch].fill(0.0);
        }
    }

    private onProcessedSamples(samples: Float32Array, finished: boolean) {
        this.firstChunkProcessed = true;
        this.processingInProgress = false;
        this.processedQueue.pushInterleaved(samples);
        // console.log(`Got new ${samples.length} samples, total ${this.processedQueueSamples}`);
        this.processingFinished = finished;
    }

    private reportFinished() {
        if (this.playbackFinished) {
            return;
        }
        this.playbackFinished = true;
        const message: PlaybackFinishedMessage = { type: 'playbackFinished' };
        this.port.postMessage(message);
    }

    private writeOutput(outputChannels: Float32Array[]): void {
        const underrun = this.processedQueue.popNonInterleaved(outputChannels);
        if (underrun > 0) {
            // Rate-limit console logging
            if (currentTime > this.lastConsoleLogTime + CONSOLE_LOG_MIN_DELTA) {
                this.lastConsoleLogTime = currentTime;
                console.error(`Underrun of ${underrun} samples`);
            }
        }
        // TODO: skipping the `underrun` samples next time?
    }

    private sendLatestSamples(numSamples: number) {
        const result = new Float32Array(numSamples * this.numChannels);

        this.processedQueue.readNonInterleaved(result);

        const message: LatestSamplesMessage = {
            type: 'latestSamples',
            samples: result,
        };
        this.port.postMessage(message);
    }

    private requestProcessSamples() {
        if (!this.client || !this.input) {
            console.error('Request processing before init()');
            return;
        }
        if (this.inputConsumed) {
            console.error('requestProcessSamples() is called when all input is consumed');
        }

        const totalSamples = this.input.length / this.numChannels;
        const availableSamples = totalSamples - this.inputOffset;
        // NOTE: We request bufferSamples at once to process. This depends on the fact that (bufferSamples / maximum
        // timeStretch) is greater than the process() output chunks size (fixed 128 for now), otherwise we would refill
        // the processed queue slower than it is consumed.
        const nextSamples = Math.min(this.bufferSamples, availableSamples);
        const nextChunk = this.input.slice(
            this.inputOffset * this.numChannels,
            (this.inputOffset + nextSamples) * this.numChannels,
        );
        this.client.processSamples(nextChunk);
        this.inputOffset += nextSamples;
        if (this.inputOffset === totalSamples) {
            this.inputConsumed = true;
            this.client.finish();
        }
        this.processingInProgress = true;
    }
}

registerProcessor(playerProcessorName, PlayerProcessor);
