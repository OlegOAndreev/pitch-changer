// RecorderProcessor is a AudioWorklet processor for recording audio. This file should only be loaded as an AudioWorklet
// module

import { recorderProcessorName } from './recorder';

const CHUNK_SIZE = 16 * 1024;

class RecorderProcessor extends AudioWorkletProcessor {
    private buffer: Float32Array = new Float32Array(CHUNK_SIZE);
    private bufferOffset: number = 0;

    constructor() {
        super();
        this.port.onmessage = (event: MessageEvent<string>) => {
            if (event.data === 'flush') {
                // Send null as a sentinel, see Recorder.start()
                this.flushBuffer();
                this.port.postMessage(null);
            }
        };
    }

    private flushBuffer(): void {
        this.port.postMessage(this.buffer.slice(0, this.bufferOffset));
        this.bufferOffset = 0;
    }

    process(inputs: Float32Array[][], _outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
        const input = inputs[0];
        if (input && input.length > 0) {
            // Assume the input is mono
            const channelData = input[0];
            if (channelData) {
                if (this.bufferOffset + channelData.length > CHUNK_SIZE) {
                    this.flushBuffer();
                }

                // Do not bother processing this case, it must be an error.
                if (channelData.length >= CHUNK_SIZE) {
                    console.error(`Channel data ${channelData.length} is greater than chunk size ${CHUNK_SIZE}, truncated`);
                }

                this.buffer.set(channelData, this.bufferOffset);
                this.bufferOffset += channelData.length;
                if (this.bufferOffset === CHUNK_SIZE) {
                    this.flushBuffer();
                }
            }
        }
        return true;
    }
}

registerProcessor(recorderProcessorName, RecorderProcessor);
