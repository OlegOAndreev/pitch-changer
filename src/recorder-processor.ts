// This is a AudioWorklet processor for Recorder.

import { recorderProcessorName, type RecordedSamplesMessage } from './recorder-types';

class RecorderProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
    }

    process(inputs: Float32Array[][], _outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
        const input = inputs[0];
        if (input && input.length > 0) {
            // Get the first channel (mono recording)
            const channelData = input[0];
            const samples = new Float32Array(channelData.length);
            samples.set(channelData);
            const message: RecordedSamplesMessage = {
                samples: samples,
            };
            this.port.postMessage(message, [samples.buffer]);
        }

        return true;
    }
}

registerProcessor(recorderProcessorName, RecorderProcessor);
