// This is a AudioWorklet processor for Player.

import { playerProcessorName, type PlayerProcessorOptions } from './player';
import { CountDownLatch, Float32RingBuffer } from './sync';

class PlayerProcessor extends AudioWorkletProcessor {
    private ringBuffer: Float32RingBuffer;
    private latch: CountDownLatch;
    private numChannels: number;
    private interleavedBuf: Float32Array | null = null;

    constructor(options: AudioWorkletNodeOptions) {
        super();
        const processorOptions = options.processorOptions as PlayerProcessorOptions;
        this.ringBuffer = new Float32RingBuffer(processorOptions.ringBufferSab);
        this.latch = new CountDownLatch(processorOptions.latchSab);
        this.numChannels = processorOptions.numChannels;
    }

    process(_inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
        const output = outputs[0];
        if (!output || output.length === 0) {
            return true;
        }

        // Assume that all channels have the same length
        const channelLength = output[0].length;
        const interleavedLength = channelLength * this.numChannels;
        // We assume outputs have always the same length and we actually will never reallocate this buffer.
        if (!this.interleavedBuf || this.interleavedBuf.length !== interleavedLength) {
            this.interleavedBuf = new Float32Array(interleavedLength);
        }

        const n = this.ringBuffer.pop(this.interleavedBuf);
        if (n !== interleavedLength) {
            if (this.ringBuffer.isClosed) {
                if (this.latch.count > 0) {
                    this.latch.countDown();
                }
                return false;
            }
            console.warn(`Ring buffer underflow: required ${interleavedLength}, got ${n}`);
        }

        // Deinterleave tempBuffer into outputs
        for (let ch = 0; ch < this.numChannels && ch < output.length; ch++) {
            const channelOut = output[ch];
            for (let i = 0; i < channelLength; i++) {
                channelOut[i] = this.interleavedBuf[i * this.numChannels + ch];
            }
        }

        // Fill any extra output channels with the first channel contents (e.g. when we play mono buffer using stereo
        // output).
        for (let ch = this.numChannels; ch < output.length; ch++) {
            const channelOut = output[ch];
            for (let i = 0; i < channelLength; i++) {
                channelOut[i] = this.interleavedBuf[i * this.numChannels];
            }
        }

        return true;
    }
}

registerProcessor(playerProcessorName, PlayerProcessor);
