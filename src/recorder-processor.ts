// This is a AudioWorklet processor for Recorder.

import { recorderProcessorName, type RecordedSamplesMessage } from './recorder-types';

class RecorderProcessor extends AudioWorkletProcessor {
    private isRecording: boolean = true;

    constructor() {
        super();
    }

    process(inputs: Float32Array[][], _outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
        if (!this.isRecording) {
            return false;
        }

        const input = inputs[0];
        if (input && input.length > 0) {
            // Get the first channel (mono recording)
            const channelData = input[0];
            const audioData = new Float32Array(channelData.length);
            audioData.set(channelData);
            const message: RecordedSamplesMessage = {
                samples: audioData,
            };
            this.port.postMessage(message, [audioData.buffer]);
        }

        return true;
    }
}

registerProcessor(recorderProcessorName, RecorderProcessor);
