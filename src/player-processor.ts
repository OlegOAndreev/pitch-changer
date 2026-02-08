// This is a AudioWorklet processor for Player.

import { playerProcessorName, type PlayerProcessorOptions } from "./player";
import { CountDownLatch, Float32RingBuffer } from "./sync";

class PlayerProcessor extends AudioWorkletProcessor {
    private ringBuffer: Float32RingBuffer;
    private latch: CountDownLatch;

    constructor(options: AudioWorkletNodeOptions) {
        super();
        const processorOptions = options.processorOptions as PlayerProcessorOptions;
        this.ringBuffer = new Float32RingBuffer(processorOptions.ringBufferSab);
        this.latch = new CountDownLatch(processorOptions.latchSab);
    }

    process(_inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
        const output = outputs[0];
        if (output && output.length > 0) {
            const n = this.ringBuffer.pop(output[0]);
            if (n !== output[0].length) {
                if (this.ringBuffer.isClosed) {
                    if (this.latch.count > 0) {
                        this.latch.countDown();
                    }
                    return false;
                }
                console.warn(`Ring buffer underflow: required ${output[0].length}, got ${n}`);
            }
            for (let ch = 1; ch < output.length; ch++) {
                // Set the same PCM dat for all channels. Assume that all channels require the same length.
                output[ch].set(output[0]);
                // output[ch].fill(0.0);
            }
        }
        return true;
    }
}

registerProcessor(playerProcessorName, PlayerProcessor);
