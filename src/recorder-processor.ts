// This is a AudioWorklet processor for Recorder.

import { recorderProcessorName, type RecorderProcessorOptions } from './recorder';
import { Float32RingBuffer } from './ring-buffer';

class RecorderProcessor extends AudioWorkletProcessor {
    private ringBuffer: Float32RingBuffer;

    constructor(options: AudioWorkletNodeOptions) {
        super();
        const ringBufferSab = (options.processorOptions as RecorderProcessorOptions).ringBufferSab;
        this.ringBuffer = new Float32RingBuffer(ringBufferSab);
    }

    process(inputs: Float32Array[][], _outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
        const input = inputs[0];
        if (input && input.length > 0) {
            const channelData = input[0];
            const pushed = this.ringBuffer.push(channelData);
            if (pushed < channelData.length && !this.ringBuffer.isClosed) {
                console.warn(`Ring buffer overflow: dropped ${channelData.length - pushed} samples`);
            }
        }
        return true;
    }
}

registerProcessor(recorderProcessorName, RecorderProcessor);
